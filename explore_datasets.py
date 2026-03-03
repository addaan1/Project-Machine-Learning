"""
=============================================================================
  EKSPLORASI & VISUALISASI DATASET
  Proyek: Prediksi Inflasi dan Dampaknya terhadap Daya Beli
  Kelompok E – Machine Learning SD-A1, Universitas Airlangga
=============================================================================

Dataset yang digunakan:
  1. Indeks Harga Konsumen / IHK  (BPS) – per kota, tahunan CSV
  2. Inflasi Bulanan M-to-M       (BPS) – per kota, tahunan CSV
  3. BI Rate / Data Inflasi       (Bank Indonesia) – bulanan Excel
  4. Upah Minimum Provinsi / UMP  (BPS) – tahunan CSV
  5. Rata-rata Pengeluaran per Kapita (BPS) – tahunan CSV

Anggota Kelompok:
  - Muhammad Rajif Al Farikhi    (162112133008)
  - Sahrul Adicandra Effendy     (164231013)
  - Semaya David Petroes Putra   (164231048)
  - Adrina Firda Marwah          (164231087)
  - Okan Athallah Maredith       (164231088)
=============================================================================
"""

import os
import glob
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import seaborn as sns
from datetime import datetime

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Kamus bulan Indonesia → angka
# ---------------------------------------------------------------------------
BULAN_MAP = {
    "Januari": 1,  "Februari": 2,  "Maret": 3,    "April": 4,
    "Mei": 5,      "Juni": 6,      "Juli": 7,     "Agustus": 8,
    "September": 9,"Oktober": 10,  "November": 11,"Desember": 12,
}


# ===========================================================================
# HELPER FUNCTIONS
# ===========================================================================

def _parse_indo_date(date_str: str) -> pd.Timestamp:
    """Parse 'Februari 2026' → Timestamp(2026-02-01)."""
    try:
        parts = str(date_str).strip().split()
        if len(parts) == 2:
            bulan = BULAN_MAP.get(parts[0], None)
            if bulan:
                return pd.Timestamp(year=int(parts[1]), month=bulan, day=1)
    except Exception:
        pass
    return pd.NaT


def _extract_year_from_filename(filename: str) -> int | None:
    """Ekstrak tahun dari nama file (4 digit terakhir sebelum .csv)."""
    try:
        stem = os.path.splitext(os.path.basename(filename))[0]
        last_part = stem.split(",")[-1].strip()
        if last_part.isdigit() and len(last_part) == 4:
            return int(last_part)
    except Exception:
        pass
    return None


def _find_indonesia_row(df: pd.DataFrame, col_idx: int = 0) -> pd.Series | None:
    """Cari baris 'INDONESIA' / 'Indonesia' di kolom tertentu."""
    col = df.iloc[:, col_idx].astype(str).str.upper().str.strip()
    match = df[col == "INDONESIA"]
    return match.iloc[0] if not match.empty else None


# ===========================================================================
# DATASET LOADERS
# ===========================================================================

def load_ikk(base_path: str) -> pd.DataFrame:
    """
    Dataset 1 – Indeks Harga Konsumen (IHK) Umum
    Sumber : BPS  |  Format : CSV per Tahun per Kota
    Kolom  : Kota, Januari, Februari, … Desember
    Skip   : 3 baris header (judul tabel BPS)
    """
    print("1. Memproses Indeks Harga Konsumen (IHK)...")
    files = glob.glob(os.path.join(base_path, "Indeks Harga Konsumen (Umum)", "*.csv"))
    records = []
    for f in sorted(files):
        tahun = _extract_year_from_filename(f)
        if not tahun:
            continue
        try:
            df = pd.read_csv(f, skiprows=3, header=0, dtype=str, on_bad_lines="skip")
            df.rename(columns={df.columns[0]: "Kota"}, inplace=True)
            row = _find_indonesia_row(df, 0)
            if row is None:
                continue
            for bulan_nama, bulan_angka in BULAN_MAP.items():
                if bulan_nama in df.columns:
                    val = str(row[bulan_nama]).replace("-", "").strip()
                    if val and val.lower() != "nan":
                        try:
                            records.append({
                                "Tanggal": pd.Timestamp(year=tahun, month=bulan_angka, day=1),
                                "IHK": float(val),
                            })
                        except ValueError:
                            pass
        except Exception as e:
            print(f"   ⚠ Gagal baca {os.path.basename(f)}: {e}")

    df_ikk = pd.DataFrame(records).sort_values("Tanggal").drop_duplicates("Tanggal").set_index("Tanggal")
    print(f"   ✓ {len(df_ikk)} observasi IHK Nasional ({df_ikk.index.min().year if not df_ikk.empty else '?'}"
          f"–{df_ikk.index.max().year if not df_ikk.empty else '?'})")
    return df_ikk


def load_inflasi_mom(base_path: str) -> pd.DataFrame:
    """
    Dataset 2 – Inflasi Bulanan M-to-M Nasional
    Sumber : BPS  |  Format : CSV per Tahun per Kota
    Kolom  : Kota, Jan, Feb, … Des  (nilai inflasi %)
    Skip   : 3 baris header BPS
    """
    print("\n2. Memproses Inflasi Bulanan (M-to-M)...")
    files = glob.glob(os.path.join(base_path, "Inflasi Bulanan", "*.csv"))
    records = []
    for f in sorted(files):
        tahun = _extract_year_from_filename(f)
        if not tahun:
            continue
        try:
            df = pd.read_csv(f, skiprows=3, header=0, dtype=str, on_bad_lines="skip")
            df.rename(columns={df.columns[0]: "Kota"}, inplace=True)
            row = _find_indonesia_row(df, 0)
            if row is None:
                continue
            for bulan_nama, bulan_angka in BULAN_MAP.items():
                if bulan_nama in df.columns:
                    val = str(row[bulan_nama]).replace("-", "").strip()
                    if val and val.lower() != "nan":
                        try:
                            records.append({
                                "Tanggal": pd.Timestamp(year=tahun, month=bulan_angka, day=1),
                                "Inflasi_MoM": float(val),
                            })
                        except ValueError:
                            pass
        except Exception as e:
            print(f"   ⚠ Gagal baca {os.path.basename(f)}: {e}")

    df_inflasi = (
        pd.DataFrame(records)
        .sort_values("Tanggal")
        .drop_duplicates("Tanggal")
        .set_index("Tanggal")
    )
    print(f"   ✓ {len(df_inflasi)} observasi Inflasi MoM Nasional ({df_inflasi.index.min().year if not df_inflasi.empty else '?'}"
          f"–{df_inflasi.index.max().year if not df_inflasi.empty else '?'})")
    return df_inflasi


def load_bi_rate(base_path: str) -> pd.DataFrame:
    """
    Dataset 3 – BI Rate / Data Inflasi Bank Indonesia
    Sumber : Bank Indonesia  |  Format : Excel (.xlsx)
    Kolom  : No, Periode (bulan-tahun indo), Data Inflasi (%)
    Skip   : 3 baris judul
    """
    print("\n3. Memproses BI Rate (Data Inflasi BI)...")
    bi_path = os.path.join(base_path, "BI Rate (Data Inflasi)", "Data Inflasi.xlsx")
    try:
        df = pd.read_excel(bi_path, skiprows=3)
        df = df[["Periode", "Data Inflasi"]].dropna()
        df["Data Inflasi"] = (
            df["Data Inflasi"].astype(str)
            .str.replace("%", "", regex=False)
            .str.strip()
        )
        df["Data Inflasi"] = pd.to_numeric(df["Data Inflasi"], errors="coerce")
        df["Tanggal"] = df["Periode"].apply(_parse_indo_date)
        df = df.dropna(subset=["Tanggal", "Data Inflasi"]).sort_values("Tanggal").set_index("Tanggal")
        print(f"   ✓ {len(df)} observasi BI Rate ({df.index.min().year}–{df.index.max().year})")
        return df[["Data Inflasi"]]
    except Exception as e:
        print(f"   ⚠ Gagal memuat BI Rate: {e}")
        return pd.DataFrame()


def load_ump(base_path: str) -> pd.DataFrame:
    """
    Dataset 4 – Upah Minimum Provinsi (UMP) per Bulan
    Sumber : BPS  |  Format : CSV per Tahun
    Kolom  : Provinsi, UMP (Rupiah)
    Skip   : 2 baris header
    """
    print("\n4. Memproses Upah Minimum Provinsi (UMP)...")
    files = glob.glob(os.path.join(base_path, "Upah Minimum Provinsi", "*.csv"))
    records = []
    for f in sorted(files):
        tahun = _extract_year_from_filename(f)
        if not tahun:
            continue
        try:
            df = pd.read_csv(f, skiprows=2, header=0, dtype=str, on_bad_lines="skip")
            df.rename(columns={df.columns[0]: "Provinsi", df.columns[1]: "UMP"}, inplace=True)
            df = df.dropna(subset=["Provinsi", "UMP"])
            df = df[~df["Provinsi"].str.strip().str.upper().isin(["PROVINSI", "INDONESIA"])]
            df["UMP"] = pd.to_numeric(df["UMP"].str.replace(",", "", regex=False), errors="coerce")
            df["Tahun"] = tahun
            records.append(df[["Provinsi", "UMP", "Tahun"]].dropna())
        except Exception as e:
            print(f"   ⚠ Gagal baca {os.path.basename(f)}: {e}")

    ump_df = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    n_tahun = ump_df["Tahun"].nunique() if not ump_df.empty else 0
    print(f"   ✓ {len(ump_df)} baris UMP dari {n_tahun} tahun ({', '.join(map(str, sorted(ump_df['Tahun'].unique()))) if not ump_df.empty else '-'})")
    return ump_df


def load_pengeluaran(base_path: str) -> pd.DataFrame:
    """
    Dataset 5 – Rata-rata Pengeluaran per Kapita Sebulan
    Sumber : BPS  |  Format : CSV per Tahun
    Kolom  : Provinsi, Perkotaan, Perdesaan, Jumlah
    Baris Nasional : 'Indonesia'
    """
    print("\n5. Memproses Rata-rata Pengeluaran per Kapita...")
    folder = "Rata-rata Pengeluaran per Kapita Sebulan Makanan dan Bukan Makanan"
    files = glob.glob(os.path.join(base_path, folder, "*.csv"))
    records = []
    for f in sorted(files):
        tahun = _extract_year_from_filename(f)
        if not tahun:
            continue
        try:
            df = pd.read_csv(f, header=0, dtype=str, on_bad_lines="skip")
            df.rename(columns={df.columns[0]: "Provinsi"}, inplace=True)
            nasional = df[df["Provinsi"].astype(str).str.strip().str.upper() == "INDONESIA"]
            if nasional.empty:
                # Fallback: ambil kolom terakhir baris pertama data
                total_col = df.columns[-1]
                nasional = df[df["Provinsi"].astype(str).str.strip().str.upper().isin(["INDONESIA", "NASIONAL"])]

            if not nasional.empty:
                # Kolom akhir = Total/Jumlah
                total_col = df.columns[-1]
                val = pd.to_numeric(
                    nasional[total_col].values[0].replace(",", "") if isinstance(nasional[total_col].values[0], str) else nasional[total_col].values[0],
                    errors="coerce",
                )
                if not pd.isna(val):
                    records.append({"Tahun": tahun, "Total_Pengeluaran": val})
        except Exception as e:
            print(f"   ⚠ Gagal baca {os.path.basename(f)}: {e}")

    pengeluaran_df = pd.DataFrame(records).sort_values("Tahun").set_index("Tahun")
    print(f"   ✓ {len(pengeluaran_df)} tahun data Pengeluaran Nasional ({', '.join(map(str, pengeluaran_df.index.tolist()))})")
    return pengeluaran_df


# ===========================================================================
# SUMMARY PRINTER
# ===========================================================================

def print_summary(datasets: dict):
    print("\n" + "=" * 70)
    print("  RINGKASAN DATASET")
    print("=" * 70)
    descs = {
        "ikk":           ("Indeks Harga Konsumen (IHK)",               "BPS",           "Bulanan / per Kota",  "IHK (angka indeks)"),
        "inflasi_mom":   ("Inflasi Bulanan M-to-M",                     "BPS",           "Bulanan / per Kota",  "Inflasi MoM (%)"),
        "bi_rate":       ("BI Rate / Data Inflasi Bank Indonesia",       "Bank Indonesia","Bulanan",             "Inflasi YoY (%)"),
        "ump":           ("Upah Minimum Provinsi (UMP)",                 "BPS",           "Tahunan / Provinsi",  "UMP (Rp/bulan)"),
        "pengeluaran":   ("Rata-rata Pengeluaran per Kapita Sebulan",    "BPS",           "Tahunan / Provinsi",  "Rupiah/kapita/bulan"),
    }
    for key, df in datasets.items():
        name, sumber, frekuensi, satuan = descs.get(key, (key, "-", "-", "-"))
        if isinstance(df, pd.DataFrame) and not df.empty:
            if isinstance(df.index, pd.DatetimeIndex):
                rentang = f"{df.index.min().strftime('%b %Y')} – {df.index.max().strftime('%b %Y')}"
            elif df.index.name == "Tahun":
                rentang = f"{df.index.min()} – {df.index.max()}"
            else:
                rentang = f"{df['Tahun'].min()} – {df['Tahun'].max()}" if "Tahun" in df.columns else "-"
            rows = len(df)
        else:
            rentang, rows = "-", 0
        print(f"\n  📦 {name}")
        print(f"     Sumber   : {sumber}")
        print(f"     Frekuensi: {frekuensi}")
        print(f"     Satuan   : {satuan}")
        print(f"     Rentang  : {rentang}")
        print(f"     Jumlah   : {rows:,} baris")
    print("\n" + "=" * 70)


# ===========================================================================
# VISUALIZATION
# ===========================================================================

COLOR_PALETTE = {
    "inflasi": "#E63946",
    "ikk":     "#4361EE",
    "bi_rate": "#3A0CA3",
    "ump":     "#2DC653",
    "pengeluaran": "#F4A261",
    "grid":    "#EEEEEE",
    "bg":      "#0D1117",
    "panel":   "#161B22",
    "text":    "#E6EDF3",
    "subtext": "#8B949E",
}


def _style_ax(ax, title: str, ylabel: str = "", xlabel: str = ""):
    ax.set_facecolor(COLOR_PALETTE["panel"])
    ax.set_title(title, color=COLOR_PALETTE["text"], fontsize=12, fontweight="bold", pad=8)
    ax.set_ylabel(ylabel, color=COLOR_PALETTE["subtext"], fontsize=9)
    ax.set_xlabel(xlabel, color=COLOR_PALETTE["subtext"], fontsize=9)
    ax.tick_params(colors=COLOR_PALETTE["subtext"], labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(COLOR_PALETTE["panel"])
    ax.yaxis.set_tick_params(labelcolor=COLOR_PALETTE["subtext"])
    ax.xaxis.set_tick_params(labelcolor=COLOR_PALETTE["subtext"])
    ax.grid(True, color=COLOR_PALETTE["grid"], alpha=0.15, linestyle="--")


def create_visualizations(datasets: dict, output_dir: str = "datasets"):
    print("\n=== MEMBUAT VISUALISASI DASHBOARD ===")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "figure.facecolor": COLOR_PALETTE["bg"],
        "text.color": COLOR_PALETTE["text"],
    })

    fig = plt.figure(figsize=(20, 14), facecolor=COLOR_PALETTE["bg"])
    fig.suptitle(
        "Dashboard Analisis Dataset: Inflasi & Daya Beli Indonesia",
        fontsize=20, fontweight="bold", color=COLOR_PALETTE["text"], y=0.98,
    )
    fig.text(
        0.5, 0.955,
        "Kelompok E – Machine Learning SD-A1, Universitas Airlangga  |  Sumber: BPS & Bank Indonesia",
        ha="center", fontsize=9, color=COLOR_PALETTE["subtext"],
    )

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35,
                           left=0.06, right=0.97, top=0.93, bottom=0.07)

    # ------------------------------------------------------------------
    # Panel 1: Inflasi Bulanan M-to-M
    # ------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    inflasi = datasets.get("inflasi_mom")
    if inflasi is not None and not inflasi.empty:
        ax1.fill_between(inflasi.index, inflasi["Inflasi_MoM"], alpha=0.25,
                         color=COLOR_PALETTE["inflasi"])
        ax1.plot(inflasi.index, inflasi["Inflasi_MoM"],
                 color=COLOR_PALETTE["inflasi"], linewidth=1.5, label="Inflasi MoM (%)")
        ax1.axhline(0, color="white", linewidth=0.5, linestyle="--", alpha=0.5)
        ax1.legend(fontsize=8, labelcolor=COLOR_PALETTE["text"])
    _style_ax(ax1, "Inflasi Bulanan Nasional (M-to-M)", ylabel="Inflasi (%)")

    # ------------------------------------------------------------------
    # Panel 2: IHK Nasional
    # ------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ikk = datasets.get("ikk")
    if ikk is not None and not ikk.empty:
        ax2.plot(ikk.index, ikk["IHK"],
                 color=COLOR_PALETTE["ikk"], linewidth=2, label="IHK Nasional")
        ax2.fill_between(ikk.index, ikk["IHK"].min(), ikk["IHK"], alpha=0.15,
                         color=COLOR_PALETTE["ikk"])
        ax2.legend(fontsize=8, labelcolor=COLOR_PALETTE["text"])
    _style_ax(ax2, "Indeks Harga Konsumen (IHK) Nasional", ylabel="Angka Indeks")

    # ------------------------------------------------------------------
    # Panel 3: BI Rate
    # ------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    bi_rate = datasets.get("bi_rate")
    if bi_rate is not None and not bi_rate.empty:
        ax3.plot(bi_rate.index, bi_rate["Data Inflasi"],
                 color=COLOR_PALETTE["bi_rate"], linewidth=2, label="Data Inflasi BI (%)")
        ax3.fill_between(bi_rate.index, bi_rate["Data Inflasi"], alpha=0.2,
                         color=COLOR_PALETTE["bi_rate"])
        ax3.legend(fontsize=8, labelcolor=COLOR_PALETTE["text"])
    _style_ax(ax3, "Data Inflasi Bank Indonesia (BI Rate)", ylabel="Inflasi (%)")

    # ------------------------------------------------------------------
    # Panel 4: UMP per Tahun (rerata nasional)
    # ------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 0])
    ump = datasets.get("ump")
    if ump is not None and not ump.empty:
        ump_avg = ump.groupby("Tahun")["UMP"].mean() / 1_000_000
        bars = ax4.bar(ump_avg.index.astype(str), ump_avg.values,
                       color=COLOR_PALETTE["ump"], alpha=0.85, width=0.5)
        for bar in bars:
            ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f"{bar.get_height():.2f}", ha="center", va="bottom",
                     fontsize=7, color=COLOR_PALETTE["subtext"])
    _style_ax(ax4, "Rata-Rata UMP Nasional per Tahun", ylabel="Juta Rupiah / Bulan")

    # ------------------------------------------------------------------
    # Panel 5: UMP Provinsi Tertinggi (tahun terakhir)
    # ------------------------------------------------------------------
    ax5 = fig.add_subplot(gs[1, 1])
    if ump is not None and not ump.empty:
        latest_year = int(ump["Tahun"].max())
        top15 = (
            ump[ump["Tahun"] == latest_year]
            .sort_values("UMP", ascending=True)
            .tail(10)
        )
        colors_bar = sns.color_palette("YlOrRd", len(top15))
        bars = ax5.barh(top15["Provinsi"], top15["UMP"] / 1_000_000, color=colors_bar, alpha=0.9)
        ax5.set_xlabel("Juta Rupiah / Bulan", color=COLOR_PALETTE["subtext"], fontsize=9)
    _style_ax(ax5, f"10 Provinsi UMP Tertinggi ({latest_year if ump is not None and not ump.empty else '-'})", ylabel="")

    # ------------------------------------------------------------------
    # Panel 6: Pengeluaran per Kapita Nasional
    # ------------------------------------------------------------------
    ax6 = fig.add_subplot(gs[1, 2])
    pengeluaran = datasets.get("pengeluaran")
    if pengeluaran is not None and not pengeluaran.empty:
        colors_p = sns.color_palette("Blues", len(pengeluaran))
        bars = ax6.bar(
            pengeluaran.index.astype(str),
            pengeluaran["Total_Pengeluaran"] / 1_000,
            color=colors_p, alpha=0.9,
        )
        ax6.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax6.tick_params(axis="x", rotation=30)
        for bar in bars:
            ax6.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                     f"{bar.get_height():,.0f}", ha="center", va="bottom",
                     fontsize=6.5, color=COLOR_PALETTE["subtext"])
    _style_ax(ax6, "Rata-rata Pengeluaran per Kapita Nasional", ylabel="Ribu Rupiah / Kapita / Bulan")

    # ------------------------------------------------------------------
    # Simpan
    # ------------------------------------------------------------------
    out_path = os.path.join(output_dir, "visualisasi_dataset.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=COLOR_PALETTE["bg"])
    plt.close()
    print(f"✓ Visualisasi disimpan → {out_path}")
    return out_path


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    datasets_dir = "datasets"

    print("=" * 70)
    print("  PROYEK: PREDIKSI INFLASI & DAMPAKNYA TERHADAP DAYA BELI")
    print("  Kelompok E – Machine Learning SD-A1, Universitas Airlangga")
    print("=" * 70 + "\n")

    print("=== MEMUAT SEMUA DATASET ===\n")
    datasets = {
        "ikk":         load_ikk(datasets_dir),
        "inflasi_mom": load_inflasi_mom(datasets_dir),
        "bi_rate":     load_bi_rate(datasets_dir),
        "ump":         load_ump(datasets_dir),
        "pengeluaran": load_pengeluaran(datasets_dir),
    }

    print_summary(datasets)
    create_visualizations(datasets, datasets_dir)

    print("\n✅ Eksplorasi dataset selesai!")


if __name__ == "__main__":
    main()