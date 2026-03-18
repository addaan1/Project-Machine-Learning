"""
=============================================================================
  PREPROCESSING PIPELINE
  Proyek: Prediksi Inflasi dan Dampaknya terhadap Daya Beli
  Kelompok E – Machine Learning SD-A1, Universitas Airlangga
=============================================================================

Output:
  1. datasets/processed/inflasi_ts.csv      → untuk Model 1 (LSTM Forecasting)
  2. datasets/processed/daya_beli_panel.csv → untuk Model 2 (Regresi Daya Beli)
=============================================================================
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

BASE = "datasets"
OUT_DIR = os.path.join(BASE, "processed")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
BULAN_MAP = {
    "Januari": 1, "Februari": 2, "Maret": 3, "April": 4,
    "Mei": 5, "Juni": 6, "Juli": 7, "Agustus": 8,
    "September": 9, "Oktober": 10, "November": 11, "Desember": 12,
}


def _parse_indo_date(s: str) -> pd.Timestamp:
    try:
        parts = str(s).strip().split()
        if len(parts) == 2:
            bulan = BULAN_MAP.get(parts[0])
            if bulan:
                return pd.Timestamp(year=int(parts[1]), month=bulan, day=1)
    except Exception:
        pass
    return pd.NaT


def _to_float_id(val) -> float:
    """Konversi angka format Indonesia (1.234,56 → 1234.56) ke float."""
    try:
        s = str(val).strip()
        # Hapus karakter non-numerik kecuali . , - dan %
        s = s.replace("%", "").replace(" ", "")
        # Format Indonesia: titik = ribuan, koma = desimal
        if "," in s and "." in s:
            s = s.replace(".", "").replace(",", ".")
        elif "," in s and "." not in s:
            s = s.replace(",", ".")
        elif "." in s and s.count(".") > 1:
            s = s.replace(".", "")
        return float(s)
    except Exception:
        return np.nan


def _extract_year(filename: str):
    try:
        stem = os.path.splitext(os.path.basename(filename))[0]
        last = stem.split(",")[-1].strip()
        if last.isdigit() and len(last) == 4:
            return int(last)
    except Exception:
        pass
    return None


def _find_indonesia(df: pd.DataFrame, col: int = 0):
    mask = df.iloc[:, col].astype(str).str.upper().str.strip() == "INDONESIA"
    return df[mask].iloc[0] if mask.any() else None


# ===========================================================================
# LOADERS — sama seperti explore_datasets.py tetapi menghasilkan nilai bersih
# ===========================================================================

def load_inflasi_mom() -> pd.DataFrame:
    """Inflasi Bulanan M-to-M → Series bulanan level INDONESIA."""
    print("  [1/7] Inflasi Bulanan M-to-M...", end=" ")
    files = glob.glob(os.path.join(BASE, "Inflasi Bulanan", "*.csv"))
    records = []
    for f in sorted(files):
        tahun = _extract_year(f)
        if not tahun:
            continue
        try:
            df = pd.read_csv(f, skiprows=3, header=0, dtype=str, on_bad_lines="skip")
            df.rename(columns={df.columns[0]: "Kota"}, inplace=True)
            row = _find_indonesia(df)
            if row is None:
                continue
            for nama, angka in BULAN_MAP.items():
                if nama in df.columns:
                    val = _to_float_id(row[nama])
                    if not np.isnan(val):
                        records.append({"Tanggal": pd.Timestamp(tahun, angka, 1),
                                        "Inflasi_MoM": val})
        except Exception:
            pass
    df_out = (pd.DataFrame(records)
              .sort_values("Tanggal")
              .drop_duplicates("Tanggal")
              .set_index("Tanggal"))
    print(f"{len(df_out)} baris ({df_out.index.min().year}–{df_out.index.max().year})")
    return df_out


def load_ihk() -> pd.DataFrame:
    """IHK Nasional (Umum) → Series bulanan."""
    print("  [2/7] Indeks Harga Konsumen (IHK)...", end=" ")
    files = glob.glob(os.path.join(BASE, "Indeks Harga Konsumen (Umum)", "*.csv"))
    records = []
    for f in sorted(files):
        tahun = _extract_year(f)
        if not tahun:
            continue
        try:
            df = pd.read_csv(f, skiprows=3, header=0, dtype=str, on_bad_lines="skip")
            df.rename(columns={df.columns[0]: "Kota"}, inplace=True)
            row = _find_indonesia(df)
            if row is None:
                continue
            for nama, angka in BULAN_MAP.items():
                if nama in df.columns:
                    val = _to_float_id(row[nama])
                    if not np.isnan(val):
                        records.append({"Tanggal": pd.Timestamp(tahun, angka, 1),
                                        "IHK": val})
        except Exception:
            pass
    df_out = (pd.DataFrame(records)
              .sort_values("Tanggal")
              .drop_duplicates("Tanggal")
              .set_index("Tanggal"))
    print(f"{len(df_out)} baris ({df_out.index.min().year}–{df_out.index.max().year})")
    return df_out


def load_bi_rate() -> pd.DataFrame:
    """BI Rate / Data Inflasi BI → Series bulanan."""
    print("  [3/7] BI Rate (Data Inflasi BI)...", end=" ")
    path = os.path.join(BASE, "BI Rate (Data Inflasi)", "Data Inflasi.xlsx")
    try:
        df = pd.read_excel(path, skiprows=3)
        df = df[["Periode", "Data Inflasi"]].dropna()
        df["Data Inflasi"] = df["Data Inflasi"].apply(_to_float_id)
        df["Tanggal"] = df["Periode"].apply(_parse_indo_date)
        df = (df.dropna(subset=["Tanggal", "Data Inflasi"])
              .sort_values("Tanggal")
              .set_index("Tanggal")
              [["Data Inflasi"]]
              .rename(columns={"Data Inflasi": "BI_Rate"}))
        print(f"{len(df)} baris ({df.index.min().year}–{df.index.max().year})")
        return df
    except Exception as e:
        print(f"GAGAL – {e}")
        return pd.DataFrame()


def load_usd_idr() -> pd.DataFrame:
    """USD/IDR kurs harian → resample ke rata-rata bulanan."""
    print("  [4/7] Kurs USD/IDR (resample bulanan)...", end=" ")
    path = os.path.join(BASE, "Data Historis USD_IDR", "Data Historis USD_IDR.csv")
    try:
        df = pd.read_csv(path, dtype=str)
        # Kolom: Tanggal, Terakhir, Pembukaan, Tertinggi, Terendah, Vol., Perubahan%
        df.rename(columns={df.columns[0]: "Tanggal", df.columns[1]: "Kurs"}, inplace=True)
        # Parse tanggal format dd/mm/yyyy
        df["Tanggal"] = pd.to_datetime(df["Tanggal"], format="%d/%m/%Y", errors="coerce")
        df["Kurs"] = df["Kurs"].apply(_to_float_id)
        df = df.dropna(subset=["Tanggal", "Kurs"]).set_index("Tanggal").sort_index()
        # Resample harian → rata-rata bulanan (period start)
        df_monthly = df[["Kurs"]].resample("MS").mean().rename(columns={"Kurs": "USD_IDR"})
        print(f"{len(df_monthly)} bulan ({df_monthly.index.min().year}–{df_monthly.index.max().year})")
        return df_monthly
    except Exception as e:
        print(f"GAGAL – {e}")
        return pd.DataFrame()


def load_ump() -> pd.DataFrame:
    """UMP per Provinsi per Tahun."""
    print("  [5/7] Upah Minimum Provinsi (UMP)...", end=" ")
    files = glob.glob(os.path.join(BASE, "Upah Minimum Provinsi", "*.csv"))
    records = []
    for f in sorted(files):
        tahun = _extract_year(f)
        if not tahun:
            continue
        try:
            df = pd.read_csv(f, skiprows=2, header=0, dtype=str, on_bad_lines="skip")
            df.rename(columns={df.columns[0]: "Provinsi", df.columns[1]: "UMP"}, inplace=True)
            df = df.dropna(subset=["Provinsi", "UMP"])
            df = df[~df["Provinsi"].str.strip().str.upper()
                    .isin(["PROVINSI", "INDONESIA", "NASIONAL"])]
            df["Provinsi"] = df["Provinsi"].str.strip().str.title()
            df["UMP"] = df["UMP"].apply(_to_float_id)
            df["Tahun"] = tahun
            records.append(df[["Provinsi", "UMP", "Tahun"]].dropna())
        except Exception:
            pass
    df_out = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    print(f"{len(df_out)} baris, {df_out['Tahun'].nunique()} tahun")
    return df_out


def load_pengeluaran() -> pd.DataFrame:
    """Rata-rata Pengeluaran per Kapita per Provinsi per Tahun."""
    print("  [6/7] Pengeluaran per Kapita...", end=" ")
    folder = "Rata-rata Pengeluaran per Kapita Sebulan Makanan dan Bukan Makanan"
    files = glob.glob(os.path.join(BASE, folder, "*.csv"))
    records = []
    for f in sorted(files):
        tahun = _extract_year(f)
        if not tahun:
            continue
        try:
            df = pd.read_csv(f, header=0, dtype=str, on_bad_lines="skip")
            df.rename(columns={df.columns[0]: "Provinsi"}, inplace=True)
            df = df[~df["Provinsi"].str.strip().str.upper()
                    .isin(["PROVINSI", ""])]
            df["Provinsi"] = df["Provinsi"].str.strip().str.title()
            # Kolom terakhir = Total/Jumlah
            total_col = df.columns[-1]
            df["Total_Pengeluaran"] = df[total_col].apply(_to_float_id)
            df["Tahun"] = tahun
            records.append(df[["Provinsi", "Total_Pengeluaran", "Tahun"]].dropna())
        except Exception:
            pass
    df_out = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    print(f"{len(df_out)} baris, {df_out['Tahun'].nunique()} tahun")
    return df_out


def load_pengangguran() -> pd.DataFrame:
    """Tingkat Pengangguran Terbuka per Provinsi per Tahun (avg Feb+Agustus)."""
    print("  [7/7] Tingkat Pengangguran Terbuka...", end=" ")
    path = os.path.join(
        BASE,
        "Tingkat Pengangguran Terbuka Berdasarkan Semester dan Provinsi di Indonesia",
        "disnakertrans-od_21012_tingkat_pengangguran_terbuka_brdsrkn_semester_prov_v1_data.csv"
    )
    try:
        df = pd.read_csv(path, dtype=str)
        df["tingkat_pengangguran_terbuka"] = df["tingkat_pengangguran_terbuka"].apply(_to_float_id)
        df["tahun"] = df["tahun"].astype(int)
        df["nama_provinsi"] = df["nama_provinsi"].str.strip().str.title()
        # Rata-rata Feb + Agustus per provinsi per tahun
        agg = (df.groupby(["nama_provinsi", "tahun"])["tingkat_pengangguran_terbuka"]
               .mean()
               .reset_index()
               .rename(columns={"nama_provinsi": "Provinsi",
                                "tahun": "Tahun",
                                "tingkat_pengangguran_terbuka": "TPT"}))
        print(f"{len(agg)} baris, {agg['Tahun'].nunique()} tahun")
        return agg
    except Exception as e:
        print(f"GAGAL – {e}")
        return pd.DataFrame()


# ===========================================================================
# BUILD OUTPUT 1: clean_inflasi_ts.csv (Tanya rekayasa, hanya raw join)
# ===========================================================================

def build_inflasi_ts(inflasi, ihk, bi_rate, usd_idr) -> pd.DataFrame:
    print("\n▶ Membangun clean_inflasi_ts.csv ...")

    # Base: inflasi MoM sebagai backbone
    ts = inflasi.copy()

    # Merge IHK (hanya ada 2005–2019, sisanya NaN)
    if not ihk.empty:
        ts = ts.join(ihk, how="left")

    # Merge BI Rate
    if not bi_rate.empty:
        ts = ts.join(bi_rate, how="left")

    # Merge USD/IDR  
    if not usd_idr.empty:
        ts = ts.join(usd_idr, how="left")

    # CATATAN REFAKTORING: 
    # Forward-fill dan Lag Features DIHAPUS dari sini untuk mencegah DATA LEAKAGE.
    # Seluruh imputasi dan windowing akan dilakukan di data_pipeline.py setelah Data Split.

    # Tambahkan fitur waktu ekstraktif (Aman dari Leakage)
    ts["Bulan"] = ts.index.month
    ts["Tahun"] = ts.index.year

    # Reset index agar Tanggal jadi kolom
    ts = ts.reset_index()

    out_path = os.path.join(OUT_DIR, "clean_inflasi_ts.csv")
    ts.to_csv(out_path, index=False)

    print(f"   ✓ {len(ts)} baris × {len(ts.columns)} kolom")
    print(f"   ✓ Rentang: {ts['Tanggal'].min().strftime('%b %Y')} – {ts['Tanggal'].max().strftime('%b %Y')}")
    print(f"   ✓ Disimpan → {out_path}")
    return ts


# ===========================================================================
# BUILD OUTPUT 2: clean_daya_beli.csv (Tanya rekayasa, hanya raw join)
# ===========================================================================

def build_daya_beli_panel(inflasi, ump, pengeluaran, pengangguran) -> pd.DataFrame:
    print("\n▶ Membangun clean_daya_beli.csv ...")

    # --- Inflasi → rata-rata tahunan ---
    inflasi_tahunan = (inflasi.reset_index()
                       .assign(Tahun=lambda x: x["Tanggal"].dt.year)
                       .groupby("Tahun")["Inflasi_MoM"]
                       .mean()
                       .reset_index()
                       .rename(columns={"Inflasi_MoM": "Inflasi_Rata_Tahunan"}))

    # --- Normalisasi nama provinsi ---
    def norm_prov(df, col="Provinsi"):
        df[col] = (df[col].str.strip()
                   .str.title()
                   .str.replace(r"\s+", " ", regex=True))
        return df

    ump_c = norm_prov(ump.copy())
    pen_c = norm_prov(pengeluaran.copy())
    tpt_c = norm_prov(pengangguran.copy())

    # --- Merge panel: Pengeluaran + UMP + TPT ---
    panel = pen_c.merge(ump_c, on=["Provinsi", "Tahun"], how="left")
    panel = panel.merge(tpt_c, on=["Provinsi", "Tahun"], how="left")
    panel = panel.merge(inflasi_tahunan, on="Tahun", how="left")

    # --- Filter tahun overlap semua variabel: 2021–2025 ---
    panel = panel[panel["Tahun"].between(2021, 2025)]

    # --- Drop baris dengan terlalu banyak NaN ---
    key_cols = ["Total_Pengeluaran", "UMP", "Inflasi_Rata_Tahunan"]
    panel = panel.dropna(subset=key_cols)

    # CATATAN REFAKTORING:
    # Transformasi LOG DIHAPUS dari sini untuk mencegah DATA LEAKAGE.
    # Transformasi log akan dipasang *setelah* train-test split di pipeline regresi.

    # --- Kolom akhir: rapikan urutan ---
    cols_order = ["Provinsi", "Tahun",
                  "Total_Pengeluaran",
                  "UMP",
                  "TPT",
                  "Inflasi_Rata_Tahunan"]
    cols_order = [c for c in cols_order if c in panel.columns]
    panel = panel[cols_order].sort_values(["Tahun", "Provinsi"]).reset_index(drop=True)

    out_path = os.path.join(OUT_DIR, "clean_daya_beli.csv")
    panel.to_csv(out_path, index=False)

    print(f"   ✓ {len(panel)} baris × {len(panel.columns)} kolom")
    print(f"   ✓ Provinsi: {panel['Provinsi'].nunique()}, Tahun: {sorted(panel['Tahun'].unique())}")
    print(f"   ✓ Disimpan → {out_path}")
    return panel


# ===========================================================================
# MAIN
# ===========================================================================

def print_summary(df: pd.DataFrame, name: str):
    print(f"\n{'─'*60}")
    print(f"  Preview: {name}")
    print(f"{'─'*60}")
    print(f"  Shape   : {df.shape}")
    print(f"  Kolom   : {list(df.columns)}")
    print(f"  Null/col: {df.isnull().sum().to_dict()}")
    print(f"  Dtypes  :")
    for col in df.columns:
        print(f"    {col:<35} {str(df[col].dtype)}")
    print(f"\n  5 baris pertama:")
    print(df.head().to_string(index=False))


def main():
    print("=" * 60)
    print("  PREPROCESSING PIPELINE – Kelompok E ML UNAIR")
    print("=" * 60)
    print("\n▶ Memuat semua dataset raw...\n")

    inflasi    = load_inflasi_mom()
    ihk        = load_ihk()
    bi_rate    = load_bi_rate()
    usd_idr    = load_usd_idr()
    ump        = load_ump()
    pengeluaran = load_pengeluaran()
    pengangguran = load_pengangguran()

    # Build outputs
    ts    = build_inflasi_ts(inflasi, ihk, bi_rate, usd_idr)
    panel = build_daya_beli_panel(inflasi, ump, pengeluaran, pengangguran)

    # Summaries
    print_summary(ts, "clean_inflasi_ts.csv")
    print_summary(panel, "clean_daya_beli.csv")

    print(f"\n{'='*60}")
    print("  ✅ Preprocessing selesai!")
    print(f"  Output disimpan di: datasets/processed/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
