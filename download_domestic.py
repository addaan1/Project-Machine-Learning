"""
============================================================================
  DOWNLOAD DOMESTIC DATASETS (Model 2: Regresi Daya Beli)
  Proyek: Prediksi Inflasi dan Dampaknya terhadap Daya Beli
============================================================================
Script ini mendownload dataset domestik per-provinsi yang relevan untuk 
analisis regresi daya beli. Karena BPS website diproteksi Cloudflare,
strategi: coba API publik, jika gagal buat template untuk diisi manual.

Output: datasets/domestic_baru/
  1. Gini_Rasio/         - Indeks Gini per Provinsi (BPS)
  2. IPM/                - Indeks Pembangunan Manusia (BPS)
  3. Garis_Kemiskinan/   - Garis Kemiskinan per Provinsi (BPS)
  4. Jumlah_Penduduk/    - Jumlah Penduduk per Provinsi (BPS)
  5. Tingkat_Urbanisasi/ - % Penduduk di Perkotaan (BPS)
  6. Inflasi_Kota/       - Inflasi per Kota (BPS)
  7. Kredit_Konsumsi/    - Kredit Konsumsi per Provinsi (Bank Indonesia)
  8. Akses_Air_Bersih/   - % Akses Air Bersih per Provinsi (BPS)
  9. Konsumsi_Protein/   - Konsumsi Protein per Kapita (BPS)
  10. Jumlah_Rumah_Tangga/ - Jumlah Rumah Tangga per Provinsi (BPS)

Setiap loader:
- Coba download dari API publik (World Bank, dll)
- Jika gagal, buat CSV template dengan kolom standar
- User tinggal download dari BPS manual & paste ke CSV
============================================================================
"""

import os
import json
import warnings
import urllib.request
import pandas as pd
import numpy as np
from datetime import datetime

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_BASE = os.path.join(BASE_DIR, "datasets", "domestic_baru")
os.makedirs(OUT_BASE, exist_ok=True)

START_YEAR = 2010
END_YEAR = 2024

# Daftar 38 provinsi Indonesia
PROVINSES = [
    "Aceh", "Sumatera Utara", "Sumatera Barat", "Riau", "Jambi",
    "Sumatera Selatan", "Bengkulu", "Lampung", "Kepulauan Bangka Belitung",
    "Kepulauan Riau", "DKI Jakarta", "Jawa Barat", "Jawa Tengah",
    "DI Yogyakarta", "Jawa Timur", "Banten", "Bali",
    "Nusa Tenggara Barat", "Nusa Tenggara Timur", "Kalimantan Barat",
    "Kalimantan Tengah", "Kalimantan Selatan", "Kalimantan Timur",
    "Kalimantan Utara", "Sulawesi Utara", "Sulawesi Tengah",
    "Sulawesi Selatan", "Sulawesi Tenggara", "Gorontalo",
    "Sulawesi Barat", "Maluku", "Maluku Utara", "Papua", "Papua Barat",
    "Papua Selatan", "Papua Tengah", "Papua Pegunungan", "Papua Daya",
]


def safe_request(url, timeout=20, headers=None):
    """Request URL dengan error handling. Return (success, content)."""
    default_headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    if headers:
        default_headers.update(headers)
    try:
        req = urllib.request.Request(url, headers=default_headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return True, resp.read()
    except Exception as e:
        return False, str(e)


def make_template(folder, filename, columns, description=""):
    """Buat template CSV kosong untuk diisi manual."""
    dir_path = os.path.join(OUT_BASE, folder)
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, filename)
    if not os.path.exists(file_path):
        # Header dengan metadata
        template = pd.DataFrame(columns=columns)
        template.to_csv(file_path, index=False)
        print(f"  [TEMPLATE] {folder}/{filename} dibuat.")
        if description:
            print(f"             {description}")
    else:
        print(f"  [EXISTS] {folder}/{filename} sudah ada, skip.")
    return file_path


def download_worldbank_indicator(country, indicator, folder, filename, col_name, 
                                  start=START_YEAR, end=END_YEAR):
    """Download indikator dari World Bank API (national level)."""
    dir_path = os.path.join(OUT_BASE, folder)
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, filename)
    
    if os.path.exists(file_path):
        print(f"  [EXISTS] {folder}/{filename} sudah ada, skip.")
        return file_path
    
    url = (f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"
           f"?date={start}:{end}&format=json&per_page=100")
    success, content = safe_request(url, timeout=30)
    
    if not success:
        print(f"  [GAGAL] World Bank API: {content[:100]}")
        return None
    
    try:
        data = json.loads(content.decode("utf-8"))
        if not isinstance(data, list) or len(data) < 2 or not data[1]:
            print(f"  [GAGAL] World Bank: data kosong")
            return None
        
        records = []
        for item in data[1]:
            if item.get("value") is not None:
                records.append({
                    "Tahun": int(item["date"]),
                    col_name: item["value"],
                })
        
        if not records:
            print(f"  [GAGAL] World Bank: tidak ada nilai")
            return None
        
        df = pd.DataFrame(records).sort_values("Tahun").reset_index(drop=True)
        df.to_csv(file_path, index=False)
        print(f"  [OK] {folder}/{filename}: {len(df)} baris "
              f"({df['Tahun'].min()}-{df['Tahun'].max()})")
        return file_path
    except Exception as e:
        print(f"  [ERROR] World Bank parse: {e}")
        return None


# ===========================================================================
# DATASET 1: INDEKS GINI per PROVINSI (BPS)
# ===========================================================================
def download_gini_rasio():
    """
    Indeks Gini Rasio per Provinsi (BPS).
    URL: https://www.bps.go.id/id/statistics-table/2/NjEjMg==/gini-rasio-menurut-provinsi.html
    Karena BPS diproteksi Cloudflare, hanya bisa manual download.
    """
    print("\n[1/10] Gini Rasio per Provinsi...")
    return make_template(
        folder="Gini_Rasio",
        filename="gini_rasio_provinsi.csv",
        columns=["Provinsi", "Tahun", "Gini_Rasio"],
        description=(
            "Format: Provinsi, Tahun, Gini_Rasio. "
            "Download manual: bps.go.id > search 'Gini Rasio Provinsi'."
        ),
    )


# ===========================================================================
# DATASET 2: IPM per PROVINSI (BPS)
# ===========================================================================
def download_ipm():
    """
    Indeks Pembangunan Manusia (IPM) per Provinsi (BPS).
    URL: https://www.bps.go.id/id/statistics-table/2/NjAjMg==/indeks-pembangunan-manusia-menurut-provinsi.html
    """
    print("\n[2/10] IPM per Provinsi...")
    return make_template(
        folder="IPM",
        filename="ipm_provinsi.csv",
        columns=["Provinsi", "Tahun", "IPM"],
        description=(
            "Format: Provinsi, Tahun, IPM. "
            "Download manual: bps.go.id > search 'IPM Provinsi'."
        ),
    )


# ===========================================================================
# DATASET 3: GARIS KEMISKINAN per PROVINSI (BPS)
# ===========================================================================
def download_garis_kemiskinan():
    """
    Garis Kemiskinan per Provinsi (BPS, Rupiah/kapita/bulan).
    URL: https://www.bps.go.id/id/statistics-table/2/NjIjMg==/garis-kemiskinan-menurut-provinsi.html
    """
    print("\n[3/10] Garis Kemiskinan per Provinsi...")
    return make_template(
        folder="Garis_Kemiskinan",
        filename="garis_kemiskinan_provinsi.csv",
        columns=["Provinsi", "Tahun", "Garis_Kemiskinan"],
        description=(
            "Format: Provinsi, Tahun, Garis_Kemiskinan (Rp/kapita/bulan). "
            "Download manual: bps.go.id > search 'Garis Kemiskinan Provinsi'."
        ),
    )


# ===========================================================================
# DATASET 4: JUMLAH PENDUDUK per PROVINSI (BPS)
# ===========================================================================
def download_jumlah_penduduk():
    """
    Jumlah Penduduk per Provinsi (BPS, hasil proyeksi Sensus Penduduk).
    URL: https://www.bps.go.id/id/statistics-table/2/NjQjMg==/jumlah-penduduk-menurut-provinsi.html
    """
    print("\n[4/10] Jumlah Penduduk per Provinsi...")
    return make_template(
        folder="Jumlah_Penduduk",
        filename="jumlah_penduduk_provinsi.csv",
        columns=["Provinsi", "Tahun", "Jumlah_Penduduk"],
        description=(
            "Format: Provinsi, Tahun, Jumlah_Penduduk. "
            "Download manual: bps.go.id > search 'Jumlah Penduduk Provinsi'."
        ),
    )


# ===========================================================================
# DATASET 5: TINGKAT URBANISASI per PROVINSI (BPS)
# ===========================================================================
def download_urbanisasi():
    """
    Persentase Penduduk di Perkotaan per Provinsi (BPS, % urban).
    URL: https://www.bps.go.id/id/statistics-table/2/NjUjMg==/persentase-penduduk-perkotaan-menurut-provinsi.html
    """
    print("\n[5/10] Tingkat Urbanisasi per Provinsi...")
    return make_template(
        folder="Tingkat_Urbanisasi",
        filename="tingkat_urbanisasi_provinsi.csv",
        columns=["Provinsi", "Tahun", "Pct_Perkotaan"],
        description=(
            "Format: Provinsi, Tahun, Pct_Perkotaan (0-100). "
            "Download manual: bps.go.id > search 'Persentase Penduduk Perkotaan'."
        ),
    )


# ===========================================================================
# DATASET 6: INFLASI KOTA per KOTA (BPS) - DIHAPUS DARI v4
# ===========================================================================
# Inflasi per kota sudah terwakili oleh dataset 'Inflasi Bulanan M-to-M'
# (data BPS #2) yang memuat seluruh kota, sehingga loader ini tidak
# diperlukan dan dihilangkan untuk menghindari duplikasi.


# ===========================================================================
# DATASET 7: KREDIT KONSUMSI per PROVINSI (Bank Indonesia) - DIHAPUS
# ===========================================================================
# Data tidak tersedia dari sumber publik (BI mempublikasi namun hanya
# untuk area tertentu, bukan per-provinsi), sehingga fitur ini dihilangkan.


# ===========================================================================
# DATASET 8: AKSES AIR BERSIH per PROVINSI (BPS)
# ===========================================================================
def download_akses_air():
    """
    Persentase Rumah Tangga dengan Akses Air Bersih (BPS, %).
    URL: https://www.bps.go.id/id/statistics-table/2/NjYjMg==/persentase-rumah-tangga-menurut-provinsi-dan-sumber-air-minum-layak.html
    """
    print("\n[8/10] Akses Air Bersih per Provinsi...")
    return make_template(
        folder="Akses_Air_Bersih",
        filename="akses_air_bersih_provinsi.csv",
        columns=["Provinsi", "Tahun", "Pct_Akses_Air_Bersih"],
        description=(
            "Format: Provinsi, Tahun, Pct_Akses_Air_Bersih. "
            "Download manual: bps.go.id > search 'Akses Air Minum Layak'."
        ),
    )


# ===========================================================================
# DATASET 9: KONSUMSI PROTEIN per KAPITA (BPS)
# ===========================================================================
def download_konsumsi_protein():
    """
    Konsumsi Protein per Kapita per Provinsi (BPS, gram/kapita/hari).
    URL: https://www.bps.go.id/id/statistics-table/2/NjMjMg==/konsumsi-protein-per-kapita.html
    """
    print("\n[9/10] Konsumsi Protein per Kapita...")
    return make_template(
        folder="Konsumsi_Protein",
        filename="konsumsi_protein_provinsi.csv",
        columns=["Provinsi", "Tahun", "Protein_gram_per_hari"],
        description=(
            "Format: Provinsi, Tahun, Protein_gram_per_hari. "
            "Download manual: bps.go.id > search 'Konsumsi Protein Provinsi'."
        ),
    )


# ===========================================================================
# DATASET 10: JUMLAH RUMAH TANGGA per PROVINSI (BPS)
# ===========================================================================
def download_jumlah_rumah_tangga():
    """
    Jumlah Rumah Tangga per Provinsi (BPS, hasil Susenas/Sensus).
    URL: https://www.bps.go.id/id/statistics-table/2/NjkjMg==/jumlah-rumah-tangga.html
    """
    print("\n[10/10] Jumlah Rumah Tangga per Provinsi...")
    return make_template(
        folder="Jumlah_Rumah_Tangga",
        filename="jumlah_rumah_tangga_provinsi.csv",
        columns=["Provinsi", "Tahun", "Jumlah_Rumah_Tangga"],
        description=(
            "Format: Provinsi, Tahun, Jumlah_Rumah_Tangga. "
            "Download manual: bps.go.id > search 'Jumlah Rumah Tangga Provinsi'."
        ),
    )


# ===========================================================================
# DATASET 11 (BONUS): DARI WORLD BANK (Nasional)
# Untuk menambah fitur yang bisa didownload otomatis
# ===========================================================================
def download_worldbank_national():
    """
    Download beberapa indikator nasional dari World Bank API.
    Ini auto-download (tidak perlu manual).
    """
    print("\n[WB API] Indikator Nasional (auto-download)...")
    
    # PPP (Purchasing Power Parity) conversion factor
    download_worldbank_indicator(
        "IDN", "PA.NUS.PPPC.RF", 
        folder="WorldBank_Nasional",
        filename="ppp_conversion_factor.csv",
        col_name="PPP_Factor",
    )
    
    # Inflation, consumer prices (annual %)
    download_worldbank_indicator(
        "IDN", "FP.CPI.TOTL.ZG", 
        folder="WorldBank_Nasional",
        filename="inflasi_worldbank.csv",
        col_name="Inflasi_WB_Annual",
    )
    
    # GDP per capita, PPP (constant 2017 international $)
    download_worldbank_indicator(
        "IDN", "NY.GDP.PCAP.PP.KD", 
        folder="WorldBank_Nasional",
        filename="gdp_percapita_ppp.csv",
        col_name="GDP_PerCapita_PPP",
    )
    
    # Unemployment (% of total labor force)
    download_worldbank_indicator(
        "IDN", "SL.UEM.TOTL.ZS", 
        folder="WorldBank_Nasional",
        filename="unemployment_worldbank.csv",
        col_name="Pct_Unemployment_WB",
    )
    
    # Poverty headcount ratio at national poverty lines (% of population)
    download_worldbank_indicator(
        "IDN", "SI.POV.NAHC", 
        folder="WorldBank_Nasional",
        filename="poverty_headcount.csv",
        col_name="Poverty_Headcount_Pct",
    )


def main():
    print("=" * 70)
    print("  DOWNLOAD DOMESTIC DATASETS (Model 2: Regresi Daya Beli)")
    print("=" * 70)
    print("  BPS website diproteksi Cloudflare → manual download.")
    print("  Template CSV akan dibuat untuk setiap dataset.")
    print("  World Bank API (nasional) → auto-download.")
    print("=" * 70)
    
    # Datasets yang harus didownload manual (BPS per provinsi)
    download_gini_rasio()
    download_ipm()
    download_garis_kemiskinan()
    download_jumlah_penduduk()
    download_urbanisasi()
    # download_inflasi_kota()  # DIHAPUS - duplikat dengan Inflasi Bulanan M-to-M
    # download_kredit_konsumsi()  # DIHAPUS - tidak ada data publik
    download_akses_air()
    download_konsumsi_protein()
    download_jumlah_rumah_tangga()
    
    # Bonus: World Bank data (auto-download)
    download_worldbank_national()
    
    print("\n" + "=" * 70)
    print("  Download selesai!")
    print(f"  Output: {OUT_BASE}")
    print("=" * 70)
    print("\n  CARA ISI MANUAL:")
    print("  1. Buka website BPS sesuai link di setiap template")
    print("  2. Download CSV/Excel")
    print("  3. Edit template CSV di folder yang sesuai:")
    print("     - Format kolom HARUS sesuai header di template")
    print("     - Nama provinsi HARUS sama dengan daftar 38 provinsi")
    print("  4. Jalankan: python preprocessing.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
