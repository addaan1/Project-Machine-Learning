# 📊 Prediksi Inflasi dan Dampaknya terhadap Daya Beli

> **Kelompok E – Machine Learning SD-A1, Universitas Airlangga**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Django](https://img.shields.io/badge/Framework-Django-green?logo=django)](https://djangoproject.com)
[![License](https://img.shields.io/badge/License-Academic-lightgrey)](LICENSE)

---

## 🎯 Deskripsi Proyek

Proyek ini membangun sistem prediksi inflasi dan analisis dampaknya terhadap daya beli masyarakat Indonesia. Terdapat dua model utama:

1. **Forecasting (LSTM)** — Memprediksi nilai inflasi bulanan ke depan berdasarkan data historis.
2. **Regresi (Random Forest / Linear Regression)** — Mengukur pengaruh inflasi terhadap daya beli masyarakat (pengeluaran per kapita).

Output disajikan melalui **Dashboard Web (Django)** yang menampilkan grafik prediksi dan fitur simulasi daya beli.

---

## 🗂️ Struktur Proyek

```
Project-Machine-Learning/
├── datasets/
│   ├── BI Rate (Data Inflasi)/
│   ├── Data Historis USD_IDR/
│   ├── Indeks Harga Konsumen (Umum)/
│   ├── Inflasi Bulanan/
│   ├── Rata-rata Pengeluaran per Kapita.../
│   ├── Tingkat Pengangguran Terbuka.../
│   ├── Upah Minimum Provinsi/
│   └── processed/
│       ├── clean_inflasi_ts.csv  ← data mentah join untuk LSTM
│       └── clean_daya_beli.csv   ← data panel daya beli mentah
├── dashboard/                    ← Django project
│   └── predictions/              ← Django app
├── explore_datasets.py           ← eksplorasi & visualisasi awal
├── preprocessing.py              ← script join dataset (menghasilkan clean_*.csv)
├── data_pipeline.py              ← ANTI-LEAKAGE PIPELINE (split, scale, log, lag)
├── requirements.txt
└── README.md
```

---

## 📦 Dataset

| # | Dataset | Sumber | Periode | Digunakan Untuk |
|---|---------|--------|---------|-----------------|
| 1 | **Indeks Harga Konsumen (IHK)** | [BPS](https://www.bps.go.id/id/statistics-table/2/MiMy/indeks-harga-konsumen--umum-.html) | 2005–2019 | Model 1 (fitur) |
| 2 | **Inflasi Bulanan (M-to-M)** | [BPS](https://www.bps.go.id/id/statistics-table/2/MSMy/inflasi--umum-.html) | 2005–2026 | Model 1 (target) |
| 3 | **Inflasi Tahun Kalender (Y-to-D)** | [BPS](https://www.bps.go.id/id/statistics-table/1/OTE0IzE=/tingkat-inflasi-harga-konsumen-nasional-tahun-kalender--y-to-d---sup-1--sup---2022-100-.html) | Historis | Referensi |
| 4 | **BI Rate / Data Inflasi** | [Bank Indonesia](https://www.bi.go.id/id/statistik/indikator/data-inflasi.aspx) | Historis | Model 1 (fitur eksogen) |
| 5 | **Upah Minimum Provinsi (UMP)** | [BPS Jateng](https://jateng.bps.go.id/id/statistics-table/2/MjgyNCMy/upah-minimum-provinsi-ump-per-bulan-menurut-provinsi-di-indonesia.html) | 2021–2025 | Model 2 (fitur) |
| 6 | **Rata-rata Pengeluaran per Kapita** | [BPS](https://www.bps.go.id/id/statistics-table/3/V1ZKMWVrSTNOek5ZZUZOcVZEZGFValJvV0hWalFUMDkjMyMwMDAw) | 2017–2025 | Model 2 (target Y) |
| 7 | **Kurs USD/IDR Historis** | [Investing.com](https://id.investing.com/currencies/usd-idr-historical-data) | Harian s/d 2024 | Model 1 (fitur eksogen) |
| 8 | **Tingkat Pengangguran Terbuka** | [Open Data Jabar](https://opendata.jabarprov.go.id/id/dataset/tingkat-pengangguran-terbuka-berdasarkan-semester-dan-provinsi-di-indonesia) | 2020–2025 | Model 2 (fitur) |

---

## 📈 Visualisasi Dataset

![Dashboard Analisis Dataset](datasets/visualisasi_dataset.png)

---

## 🔧 Preprocessing & Data Pipeline (Anti-Leakage)

Proses pengolahan data dibagi menjadi dua tahapan ketat untuk **mencegah Data Leakage** dari *testing set* ke *training set*:

1. **`preprocessing.py`**: Hanya melakukan pembersihan teks dan penggabungan secara waktu (join).
2. **`data_pipeline.py`**: Melakukan Train/Val/Test Split *TERLEBIH DAHULU*, kemudian melakukan *Scaling*, Interpolasi, Log Transform, dan pembuatan fitur *Lag/Windows*.

---

### Output 1 — `datasets/processed/clean_inflasi_ts.csv` (Raw untuk LSTM)

**Alur `preprocessing.py`:**
```text
Inflasi Bulanan (22 file CSV, 2005–2026)
  → Parse tanggal bahasa Indonesia
  → Gabungkan jadi 1 kolom: [Tanggal, Inflasi_MoM]
  → Join IHK (NaN untuk data setelah 2019)
  → Join BI Rate (bulanan)
  → Join USD/IDR (resample harian → bulanan)
  → Tambah kolom Bulan dan Tahun
```
*(Catatan: Fitur lag 1-12 dan scaling akan digenerate otomatis di memori oleh `data_pipeline.py` spesifik pada data Train untuk mencegah leakage)*.

| Kolom | Keterangan |
|-------|-----------|
| `Tanggal` | Periode bulanan (2005–2026) |
| `Inflasi_MoM` | Target prediksi (%) |
| `IHK` | Indeks harga konsumen (NaN setelah 2019) |
| `USD_IDR` | Rata-rata kurs bulanan (Rp) |
| `BI_Rate` | Suku bunga acuan BI (%) |
| `Bulan`, `Tahun` | Fitur siklus waktu |

---

### Output 2 — `datasets/processed/clean_daya_beli.csv` (Raw untuk Regresi)

**Alur `preprocessing.py`:**
```text
Pengeluaran per Kapita (per provinsi, 2017–2025)
  → Join UMP per provinsi (2021–2025)
  → Join Tingkat Pengangguran Terbuka per provinsi (2020–2025)
  → Join Inflasi rata-rata tahunan (dari inflasi bulanan)
  → Filter tahun overlap: 2021–2025
```

| Kolom | Keterangan |
|-------|-----------|
| `Provinsi` | 38 provinsi Indonesia |
| `Tahun` | 2021–2025 |
| `Total_Pengeluaran` | Pengeluaran per kapita (Rp/bulan) — **Target Y** |
| `UMP` | Upah minimum (Rp/bulan) |
| `TPT` | Tingkat Pengangguran Terbuka (%) |
| `Inflasi_Rata_Tahunan` | Rata-rata inflasi MoM per tahun (%) |

---

## 🤖 Model Machine Learning

### Model 1 – Forecasting Inflasi (LSTM)
- **Input**: *Windowing sequences* 12 bulanan (`clean_inflasi_ts.csv` diproses oleh `data_pipeline.py`)
- **Output**: Prediksi inflasi bulan berikutnya
- **Metrik**: MAE, RMSE

### Model 2 – Dampak Inflasi terhadap Daya Beli (Regresi)
- **Input**: Panel data provinsi (`clean_daya_beli.csv` diproses oleh `data_pipeline.py`)
- **Output**: Estimasi pengeluaran per kapita berdasarkan inflasi & variabel ekonomi lainnya
- **Metrik**: R², MSE, koefisien regresi

---

## 🚀 Cara Menjalankan

```bash
# 1. Install dependensi
pip install -r requirements.txt

# 2. Eksplorasi dataset (opsional)
python explore_datasets.py

# 3. Jalankan preprocessing (Menghasilkan clean_*.csv)
python preprocessing.py

# 4. Tes Split Pipeline AI
python data_pipeline.py

# 5. Jalankan web dashboard
cd dashboard
python manage.py runserver
```

---

## 👥 Anggota Kelompok E

| Nama | NIM |
|------|-----|
| Muhammad Rajif Al Farikhi | 162112133008 |
| Sahrul Adicandra Effendy | 164231013 |
| Semaya David Petroes Putra | 164231048 |
| Adrina Firda Marwah | 164231087 |
| Okan Athallah Maredith | 164231088 |

---

## 📚 Referensi Data
- Badan Pusat Statistik (BPS): https://www.bps.go.id
- Bank Indonesia: https://www.bi.go.id
- Open Data Jabar: https://opendata.jabarprov.go.id
- Investing.com: https://id.investing.com
