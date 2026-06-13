import os
import re
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

DATA_DIR = "datasets"
OUT_DIR = os.path.join(DATA_DIR, 'processed')

PROVINSI_NORMALIZE = {
    'ACEH': 'Aceh',
    'SUMATERA UTARA': 'Sumatera Utara', 'SUMATERA BARAT': 'Sumatera Barat',
    'SUMATERA SELATAN': 'Sumatera Selatan', 'RIAU': 'Riau',
    'JAMBI': 'Jambi', 'BENGKULU': 'Bengkulu', 'LAMPUNG': 'Lampung',
    'KEPULAUAN BANGKA BELITUNG': 'Kepulauan Bangka Belitung',
    'KEPULAUAN RIAU': 'Kepulauan Riau',
    'DKI JAKARTA': 'DKI Jakarta',
    'JAWA BARAT': 'Jawa Barat', 'JAWA TENGAH': 'Jawa Tengah',
    'DAERAH ISTIMEWA YOGYAKARTA': 'DI Yogyakarta',
    'JAWA TIMUR': 'Jawa Timur', 'BANTEN': 'Banten',
    'BALI': 'Bali',
    'NUSA TENGGARA BARAT': 'Nusa Tenggara Barat',
    'NUSA TENGGARA TIMUR': 'Nusa Tenggara Timur',
    'KALIMANTAN BARAT': 'Kalimantan Barat', 'KALIMANTAN TENGAH': 'Kalimantan Tengah',
    'KALIMANTAN SELATAN': 'Kalimantan Selatan', 'KALIMANTAN TIMUR': 'Kalimantan Timur',
    'KALIMANTAN UTARA': 'Kalimantan Utara',
    'SULAWESI UTARA': 'Sulawesi Utara', 'SULAWESI TENGAH': 'Sulawesi Tengah',
    'SULAWESI SELATAN': 'Sulawesi Selatan', 'SULAWESI TENGGARA': 'Sulawesi Tenggara',
    'SULAWESI BARAT': 'Sulawesi Barat', 'GORONTALO': 'Gorontalo',
    'MALUKU': 'Maluku', 'MALUKU UTARA': 'Maluku Utara',
    'PAPUA BARAT': 'Papua Barat', 'PAPUA': 'Papua',
    'Papua Selatan': 'Papua', 'Papua Tengah': 'Papua', 'Papua Pegunungan': 'Papua',
    'Indonesia': None,  # Exclude national aggregate
}

VALID_PROVINSI = [
    'Aceh', 'Sumatera Utara', 'Sumatera Barat', 'Sumatera Selatan', 'Riau',
    'Jambi', 'Bengkulu', 'Lampung', 'Kepulauan Bangka Belitung', 'Kepulauan Riau',
    'DKI Jakarta', 'Jawa Barat', 'Jawa Tengah', 'DI Yogyakarta', 'Jawa Timur',
    'Banten', 'Bali', 'Nusa Tenggara Barat', 'Nusa Tenggara Timur',
    'Kalimantan Barat', 'Kalimantan Tengah', 'Kalimantan Selatan',
    'Kalimantan Timur', 'Kalimantan Utara',
    'Sulawesi Utara', 'Sulawesi Tengah', 'Sulawesi Selatan', 'Sulawesi Tenggara',
    'Sulawesi Barat', 'Gorontalo', 'Maluku', 'Maluku Utara',
    'Papua Barat', 'Papua',
]

def normalize_provinsi(name):
    if pd.isna(name):
        return None
    s = str(name).strip()
    s_upper = s.upper()
    if s_upper in PROVINSI_NORMALIZE:
        result = PROVINSI_NORMALIZE[s_upper]
        if result is None:
            return None
        return result
    # Check case-insensitive
    for k, v in PROVINSI_NORMALIZE.items():
        if k.upper() == s_upper:
            return v
    # Try matching title case
    for p in VALID_PROVINSI:
        if p.lower() == s.lower():
            return p
    return None

def extract_year_from_filename(name):
    matches = re.findall(r'(\d{4})', name)
    return int(matches[-1]) if matches else None

def load_ump():
    base = os.path.join(DATA_DIR, 'Upah Minimum Provinsi')
    rows = []
    for f in sorted(os.listdir(base)):
        year = extract_year_from_filename(f)
        if year is None:
            continue
        try:
            df = pd.read_csv(os.path.join(base, f), skiprows=1)
            df.columns = ['Provinsi', 'UMP']
            df['Tahun'] = year
            df['Provinsi'] = df['Provinsi'].apply(normalize_provinsi)
            df = df[df['Provinsi'].isin(VALID_PROVINSI)][['Tahun', 'Provinsi', 'UMP']]
            df['UMP'] = pd.to_numeric(df['UMP'], errors='coerce')
            rows.append(df)
        except Exception as e:
            print(f"UMP {f}: {e}")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def load_tpt():
    base = os.path.join(DATA_DIR, 'Tingkat Pengangguran Terbuka (TPT) dan Tingkat Partisipasi Angkatan Kerja (TPAK) Menurut Provinsi')
    rows = []
    for f in sorted(os.listdir(base)):
        year = extract_year_from_filename(f)
        if year is None:
            continue
        try:
            df = pd.read_csv(os.path.join(base, f))
            tpt_cols = [c for c in df.columns if 'TPT' in c and 'TPAK' not in c]
            tpak_cols = [c for c in df.columns if 'TPAK' in c]
            if not tpt_cols:
                continue
            # Handle concatenated values
            tpt_data = df[tpt_cols].copy()
            for col in tpt_cols:
                if tpt_data[col].dtype == 'object':
                    def parse_val(v):
                        if pd.isna(v): return np.nan
                        s = str(v)
                        if '.' in s and len(s.split('.')) == 3 and len(s.split('.')[2]) == 2:
                            parts = s.split('.')
                            return (float(parts[0] + '.' + parts[1][:2]) + float(parts[1][2:] + '.' + parts[2])) / 2
                        try: return float(s)
                        except: return np.nan
                    tpt_data[col] = pd.to_numeric(tpt_data[col].apply(parse_val), errors='coerce')
                else:
                    tpt_data[col] = pd.to_numeric(tpt_data[col], errors='coerce')
            df['TPT'] = tpt_data.mean(axis=1)

            if tpak_cols:
                tpak_data = df[tpak_cols].copy()
                for col in tpak_cols:
                    if tpak_data[col].dtype == 'object':
                        def parse_val(v):
                            if pd.isna(v): return np.nan
                            s = str(v)
                            if '.' in s and len(s.split('.')) == 3 and len(s.split('.')[2]) == 2:
                                parts = s.split('.')
                                return (float(parts[0] + '.' + parts[1][:2]) + float(parts[1][2:] + '.' + parts[2])) / 2
                            try: return float(s)
                            except: return np.nan
                        tpak_data[col] = pd.to_numeric(tpak_data[col].apply(parse_val), errors='coerce')
                df['TPAK'] = tpak_data.mean(axis=1)

            df['Tahun'] = year
            df['Provinsi'] = df['Provinsi'].apply(normalize_provinsi)
            df = df[df['Provinsi'].isin(VALID_PROVINSI)]
            keep = ['Tahun', 'Provinsi', 'TPT']
            if 'TPAK' in df.columns: keep.append('TPAK')
            df = df[keep]
            rows.append(df)
        except Exception as e:
            print(f"TPT {f}: {e}")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def load_pdrb():
    base = os.path.join(DATA_DIR, 'Produk Domestik Regional Bruto Per Kapita (Ribu Rupiah)')
    rows = []
    for f in sorted(os.listdir(base)):
        year = extract_year_from_filename(f)
        if year is None or year < 2017:
            continue
        try:
            df = pd.read_csv(os.path.join(base, f), skiprows=3)
            first_col = df.columns[0]
            df = df.rename(columns={first_col: 'Provinsi'})
            year_cols = [c for c in df.columns if c != 'Provinsi' and str(year) in str(c)]
            if len(year_cols) < 2: continue
            df = df[['Provinsi', year_cols[0], year_cols[1]]].copy()
            df.columns = ['Provinsi', 'PDRB_HargaBerlaku', 'PDRB_HargaKonstan']
            df['Tahun'] = year
            df['Provinsi'] = df['Provinsi'].apply(normalize_provinsi)
            df = df[df['Provinsi'].isin(VALID_PROVINSI)]
            df['PDRB_HargaBerlaku'] = pd.to_numeric(df['PDRB_HargaBerlaku'], errors='coerce')
            df['PDRB_HargaKonstan'] = pd.to_numeric(df['PDRB_HargaKonstan'], errors='coerce')
            rows.append(df)
        except Exception as e:
            print(f"PDRB {f}: {e}")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def load_pengeluaran():
    base = os.path.join(DATA_DIR, 'Rata-rata Pengeluaran per Kapita Sebulan Makanan dan Bukan Makanan')
    rows = []
    for f in sorted(os.listdir(base)):
        year = extract_year_from_filename(f)
        if year is None or year < 2017:
            continue
        try:
            df = pd.read_csv(os.path.join(base, f))
            cols = df.columns.tolist()
            makan_col = next((c for c in cols if 'Makanan' in c), None)
            bukan_col = next((c for c in cols if 'Bukan Makanan' in c), None)
            total_col = next((c for c in cols if 'Jumlah' in c), None)
            if not total_col: continue
            df = df.rename(columns={
                makan_col: 'Pengeluaran_Makanan' if makan_col else None,
                bukan_col: 'Pengeluaran_Bukan_Makanan' if bukan_col else None,
                total_col: 'Total_Pengeluaran'
            })
            df['Tahun'] = year
            df['Provinsi'] = df['Provinsi'].apply(normalize_provinsi)
            df = df[df['Provinsi'].isin(VALID_PROVINSI)]
            keep = ['Tahun', 'Provinsi', 'Total_Pengeluaran']
            if 'Pengeluaran_Makanan' in df.columns: keep.append('Pengeluaran_Makanan')
            if 'Pengeluaran_Bukan_Makanan' in df.columns: keep.append('Pengeluaran_Bukan_Makanan')
            df = df[keep]
            for c in keep[2:]:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            rows.append(df)
        except Exception as e:
            print(f"Pengeluaran {f}: {e}")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def load_inflasi_tahunan():
    path = os.path.join(DATA_DIR, 'Inflasi Bulanan')
    rows = []
    for f in sorted(os.listdir(path)):
        year = extract_year_from_filename(f)
        if year is None or year < 2017:
            continue
        try:
            for skip in range(5):
                df_try = pd.read_csv(os.path.join(path, f), skiprows=skip, nrows=2)
                cols = df_try.columns.tolist()
                if 'Bulan' in cols or 'Tahunan' in cols:
                    df = pd.read_csv(os.path.join(path, f), skiprows=skip)
                    break
            else:
                continue

            if 'Bulan' in df.columns and 'Data Inflasi' in df.columns:
                df = df[['Bulan', 'Data Inflasi']].copy()
                df.columns = ['Bulan', 'Inflasi_MoM']
                df['Inflasi_MoM'] = pd.to_numeric(df['Inflasi_MoM'], errors='coerce')
                annual = pd.DataFrame([{'Tahun': year, 'Inflasi_Rata_Tahunan': df['Inflasi_MoM'].mean()}])
                rows.append(annual)
            elif 'Tahunan' in df.columns:
                nasional_row = df[df.iloc[:, 0].astype(str).str.contains('NASIONAL|Nasional|nasional', na=False)]
                if not nasional_row.empty:
                    tahunan_val = nasional_row['Tahunan'].iloc[0]
                else:
                    tahunan_val = pd.to_numeric(df['Tahunan'], errors='coerce').mean()
                rows.append(pd.DataFrame([{'Tahun': year, 'Inflasi_Rata_Tahunan': tahunan_val}]))
        except Exception as e:
            print(f"Inflasi {f}: {e}")
    if not rows: return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)

def load_worldbank():
    base = os.path.join(DATA_DIR, 'domestic_baru', 'WorldBank_Nasional')
    mapping = {
        'unemployment_worldbank.csv': 'Pct_Unemployment_WB',
        'poverty_headcount.csv': 'Poverty_Headcount_Pct',
        'inflasi_worldbank.csv': 'Inflasi_WB_Annual',
        'gdp_percapita_ppp.csv': 'GDP_PerCapita_PPP',
    }
    rows = []
    for f, col_name in mapping.items():
        path = os.path.join(base, f)
        if not os.path.exists(path): continue
        try:
            df = pd.read_csv(path)
            year_col = [c for c in df.columns if 'date' in c.lower() or 'year' in c.lower() or 'tahun' in c.lower()][0]
            val_col = [c for c in df.columns if c != year_col][0]
            df = df[[year_col, val_col]].copy()
            df.columns = ['Tahun', col_name]
            df['Tahun'] = pd.to_numeric(df['Tahun'], errors='coerce')
            df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
            df = df.dropna()
            df['Tahun'] = df['Tahun'].astype(int)
            rows.append(df)
        except Exception as e:
            print(f"WB {f}: {e}")
    if not rows: return pd.DataFrame()
    out = rows[0]
    for r in rows[1:]:
        out = out.merge(r, on='Tahun', how='outer')
    return out

# Load
print("Loading...")
df_ump = load_ump()
df_tpt = load_tpt()
df_pdrb = load_pdrb()
df_peng = load_pengeluaran()
df_inf = load_inflasi_tahunan()
df_wb = load_worldbank()

print(f"UMP: {df_ump.shape} | {(df_ump['Tahun'].unique()) if len(df_ump) else 'empty'}")
print(f"TPT: {df_tpt.shape} | {(df_tpt['Tahun'].unique()) if len(df_tpt) else 'empty'}")
print(f"PDRB: {df_pdrb.shape} | {(df_pdrb['Tahun'].unique()) if len(df_pdrb) else 'empty'}")
print(f"Peng: {df_peng.shape} | {(df_peng['Tahun'].unique()) if len(df_peng) else 'empty'}")
print(f"Inf: {df_inf.shape}")
print(f"WB: {df_wb.shape}")

# Drop duplicates within each dataset
for name, df in [('UMP', df_ump), ('TPT', df_tpt), ('PDRB', df_pdrb), ('Peng', df_peng)]:
    before = len(df)
    df.drop_duplicates(subset=['Tahun', 'Provinsi'], inplace=True)
    print(f"  {name} dedup: {before} -> {len(df)}")

# Merge
print("\nMerging...")
df = df_peng.merge(df_ump, on=['Tahun', 'Provinsi'], how='outer')
df = df.merge(df_tpt, on=['Tahun', 'Provinsi'], how='outer')
df = df.merge(df_pdrb, on=['Tahun', 'Provinsi'], how='outer')
df = df.merge(df_inf, on='Tahun', how='left')
df = df.merge(df_wb, on='Tahun', how='left')

print(f"After merge: {df.shape}")
print(f"Years: {sorted(df['Tahun'].dropna().unique())}")
print(f"Provinsi: {df['Provinsi'].nunique()}")
print(f"Rows per year:")
print(df.groupby('Tahun').size().to_string())

# Drop duplicates after merge
before = len(df)
df.drop_duplicates(subset=['Tahun', 'Provinsi'], inplace=True)
print(f"After final dedup: {before} -> {len(df)}")

# Forward/backward fill within province
df = df.sort_values(['Provinsi', 'Tahun']).reset_index(drop=True)
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in num_cols:
    if col == 'Tahun': continue
    df[col] = df.groupby('Provinsi')[col].transform(lambda x: x.ffill().bfill())

print(f"Null after fill: {df.isnull().sum().sum()}")

# Drop rows without target
df = df.dropna(subset=['Total_Pengeluaran'])
print(f"Final shape: {df.shape}")

# Save
out_path = os.path.join(OUT_DIR, 'clean_daya_beli.csv')
df.to_csv(out_path, index=False)
print(f"\nSAVED: {out_path}")
print(df.groupby('Tahun').size().to_string())
