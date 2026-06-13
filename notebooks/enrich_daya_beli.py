"""
Add IPM, Jumlah_Penduduk, Pct_Populasi, Gini_Rasio, Garis_Kemiskinan,
Akses_Air_Bersih, Protein ke clean_daya_beli.csv.
"""
import os
import re
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

DATA_DIR = "datasets"
OUT_DIR = os.path.join(DATA_DIR, 'processed')

def extract_year(name):
    matches = re.findall(r'(\d{4})', name)
    return int(matches[-1]) if matches else None

def normalize_provinsi(name):
    if pd.isna(name): return None
    s = str(name).strip().upper()
    mapping = {
        'ACEH': 'Aceh', 'SUMATERA UTARA': 'Sumatera Utara',
        'SUMATERA BARAT': 'Sumatera Barat', 'SUMATERA SELATAN': 'Sumatera Selatan',
        'RIAU': 'Riau', 'KEPULAUAN RIAU': 'Kepulauan Riau',
        'JAMBI': 'Jambi', 'BENGKULU': 'Bengkulu', 'LAMPUNG': 'Lampung',
        'KEPULAUAN BANGKA BELITUNG': 'Kepulauan Bangka Belitung',
        'DKI JAKARTA': 'DKI Jakarta', 'JAWA BARAT': 'Jawa Barat',
        'JAWA TENGAH': 'Jawa Tengah', 'JAWA TIMUR': 'Jawa Timur',
        'DAERAH ISTIMEWA YOGYAKARTA': 'DI Yogyakarta',
        'BANTEN': 'Banten', 'BALI': 'Bali',
        'NUSA TENGGARA BARAT': 'Nusa Tenggara Barat',
        'NUSA TENGGARA TIMUR': 'Nusa Tenggara Timur',
        'KALIMANTAN BARAT': 'Kalimantan Barat',
        'KALIMANTAN TENGAH': 'Kalimantan Tengah',
        'KALIMANTAN SELATAN': 'Kalimantan Selatan',
        'KALIMANTAN TIMUR': 'Kalimantan Timur',
        'KALIMANTAN UTARA': 'Kalimantan Utara',
        'SULAWESI UTARA': 'Sulawesi Utara', 'SULAWESI TENGAH': 'Sulawesi Tengah',
        'SULAWESI SELATAN': 'Sulawesi Selatan', 'SULAWESI TENGGARA': 'Sulawesi Tenggara',
        'SULAWESI BARAT': 'Sulawesi Barat', 'GORONTALO': 'Gorontalo',
        'MALUKU': 'Maluku', 'MALUKU UTARA': 'Maluku Utara',
        'PAPUA BARAT': 'Papua Barat', 'PAPUA': 'Papua',
    }
    return mapping.get(s, None if 'INDONESIA' in s else s.title())

def load_folder_data(folder, value_col_name, skiprows=0, has_year_in_name=True):
    """Load all CSV files in a folder, each is one year."""
    full = os.path.join(DATA_DIR, 'domestic_baru', folder)
    if not os.path.exists(full):
        print(f"Folder not found: {full}")
        return pd.DataFrame()
    rows = []
    for f in sorted(os.listdir(full)):
        if not f.endswith('.csv'):
            continue
        path = os.path.join(full, f)
        year = extract_year(f)
        if year is None:
            continue
        try:
            df = pd.read_csv(path, skiprows=skiprows)
            # First col is Provinsi
            if 'Provinsi' in df.columns:
                prov_col = 'Provinsi'
            else:
                prov_col = df.columns[0]
            # Find value column (numeric, not year)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                continue
            # Take first numeric col
            val_col = numeric_cols[0]
            df = df[[prov_col, val_col]].copy()
            df.columns = ['Provinsi', value_col_name]
            df['Tahun'] = year
            df['Provinsi'] = df['Provinsi'].apply(normalize_provinsi)
            df = df[df['Provinsi'].notna()]
            df[value_col_name] = pd.to_numeric(df[value_col_name], errors='coerce')
            rows.append(df)
        except Exception as e:
            print(f"  {folder}/{f}: {e}")
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)

# Load base panel
df_base = pd.read_csv(os.path.join(OUT_DIR, 'clean_daya_beli.csv'))
print(f"Base panel: {df_base.shape}")

# Load additional features
print("\nLoading additional features...")
sources = [
    ('IPM', 'IPM', 0),
    ('Jumlah_Penduduk', 'Jumlah_Penduduk', 0),
    ('Tingkat_Urbanisasi', 'Pct_Populasi', 0),
    ('Gini_Rasio', 'Gini_Rasio', 0),
    ('Garis_Kemiskinan', 'Garis_Kemiskinan', 0),
    ('Akses_Air_Bersih', 'Pct_Akses_Air_Bersih', 0),
    ('Konsumsi_Protein', 'Protein_gram_per_hari', 0),
]

for folder, col, skip in sources:
    df_new = load_folder_data(folder, col, skiprows=skip)
    if len(df_new) > 0:
        years_list = sorted(df_new['Tahun'].unique())
        print(f'  {col}: {len(df_new)} rows | years: {years_list}')
        # Merge
        df_base = df_base.merge(df_new, on=['Tahun', 'Provinsi'], how='left')
    else:
        print(f'  {col}: no data')

print(f"\nFinal panel: {df_base.shape}")
print(f"Null counts:")
nulls = df_base.isnull().sum()
print(nulls[nulls > 0])

# Forward/backward fill within province for new features
df_base = df_base.sort_values(['Provinsi', 'Tahun']).reset_index(drop=True)
for col in df_base.columns:
    if col in ['Tahun', 'Provinsi', 'Total_Pengeluaran']:
        continue
    if df_base[col].dtype in [np.float64, np.int64]:
        df_base[col] = df_base.groupby('Provinsi')[col].transform(lambda x: x.ffill().bfill())

print(f"Null after fill: {df_base.isnull().sum().sum()}")

# Save
df_base.to_csv(os.path.join(OUT_DIR, 'clean_daya_beli.csv'), index=False)
print(f"\nSAVED: {df_base.shape}")
