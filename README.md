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
│       ├── inflasi_ts.csv        ← input Model 1 (LSTM)
│       └── daya_beli_panel.csv   ← input Model 2 (Regresi)
├── dashboard/                    ← Django project
│   └── predictions/              ← Django app
├── explore_datasets.py           ← eksplorasi & visualisasi awal
├── preprocessing.py              ← pipeline pembersihan data
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

## 🔧 Preprocessing Pipeline

Sebelum masuk ke modeling, semua dataset mentah diproses menggunakan **`preprocessing.py`**. Berikut penjelasan tiap tahapan:

### Masalah yang Ditemukan di Data Mentah

| Masalah | Contoh | Solusi |
|---------|--------|--------|
| Format angka Indonesia | `15.500,00` | Konversi: hapus titik ribuan, ganti koma → titik |
| Tanda persen di teks | `"2,5%"` | Strip `%`, konversi ke `float` |
| Tanggal bahasa Indonesia | `"Februari 2026"` | Parse manual dengan kamus bulan |
| Data harian perlu dibulatkan | USD/IDR harian | Resample ke rata-rata bulanan (`resample('MS').mean()`) |
| Perbedaan frekuensi antar dataset | Tahunan vs Bulanan | Agregasi inflasi ke rata-rata tahunan untuk model regresi |
| NaN pada data IHK post-2019 | IHK hanya tersedia 2005–2019 | Dibiarkan NaN (di-handle LSTM dengan masking) |
| Baris non-data di header BPS | 3 baris judul tabel | `skiprows=3` saat membaca CSV |

---

### Output 1 — `datasets/processed/inflasi_ts.csv` (untuk Model 1 LSTM)

**Alur:**
```
Inflasi Bulanan (22 file CSV, 2005–2026)
  → Parse tanggal bahasa Indonesia
  → Filter baris "INDONESIA"
  → Gabungkan jadi 1 kolom: [Tanggal, Inflasi_MoM]
  → Join IHK (NaN untuk data setelah 2019)
  → Join BI Rate (bulanan)
  → Join USD/IDR (resample harian → bulanan, lalu ffill)
  → Buat lag features: Inflasi_MoM_lag1 s/d lag12
  → Tambah kolom Bulan dan Tahun
  → Drop 12 baris pertama (lag belum terisi)
```

**Hasil:** `242 baris × 18 kolom`

| Kolom | Keterangan |
|-------|-----------|
| `Tanggal` | Periode bulanan (2006–2026) |
| `Inflasi_MoM` | Target prediksi (%) |
| `IHK` | Indeks harga konsumen (NaN setelah 2019) |
| `USD_IDR` | Rata-rata kurs bulanan (Rp) |
| `BI_Rate` | Suku bunga acuan BI (%) |
| `Inflasi_MoM_lag1` … `lag12` | Nilai inflasi 1–12 bulan sebelumnya |
| `Bulan`, `Tahun` | Fitur siklus waktu |

---

### Output 2 — `datasets/processed/daya_beli_panel.csv` (untuk Model 2 Regresi)

**Alur:**
```
Pengeluaran per Kapita (per provinsi, 2017–2025)
  → Join UMP per provinsi (2021–2025)
  → Join Tingkat Pengangguran Terbuka per provinsi (2020–2025)
  → Join Inflasi rata-rata tahunan (dari inflasi bulanan)
  → Filter tahun overlap: 2021–2025
  → Drop baris dengan kolom utama kosong
  → Transformasi log: log(Total_Pengeluaran), log(UMP)
```

**Hasil:** `177 baris × 8 kolom` (38 provinsi × 5 tahun = 190, minus 13 data Papua kosong)

| Kolom | Keterangan |
|-------|-----------|
| `Provinsi` | 38 provinsi Indonesia |
| `Tahun` | 2021–2025 |
| `Total_Pengeluaran` | Pengeluaran per kapita (Rp/bulan) — **Target Y** |
| `log_Total_Pengeluaran` | Transformasi log untuk normalisasi |
| `UMP` | Upah minimum (Rp/bulan) |
| `log_UMP` | Transformasi log untuk normalisasi |
| `TPT` | Tingkat Pengangguran Terbuka (%) |
| `Inflasi_Rata_Tahunan` | Rata-rata inflasi MoM per tahun (%) |

---

## 🤖 Model Machine Learning

### Model 1 – Forecasting Inflasi (LSTM)
- **Input**: Sequence 12 bulan terakhir (`inflasi_ts.csv`)
- **Output**: Prediksi inflasi bulan berikutnya
- **Metrik**: MAE, RMSE

### Model 2 – Dampak Inflasi terhadap Daya Beli (Regresi)
- **Input**: Panel data provinsi (`daya_beli_panel.csv`)
- **Output**: Estimasi pengeluaran per kapita berdasarkan inflasi & variabel ekonomi lainnya
- **Metrik**: R², MSE, koefisien regresi

---

## 🚀 Cara Menjalankan

```bash
# 1. Install dependensi
pip install -r requirements.txt

# 2. Eksplorasi dataset (opsional)
python explore_datasets.py

# 3. Jalankan preprocessing
python preprocessing.py
# → Output: datasets/processed/inflasi_ts.csv
# → Output: datasets/processed/daya_beli_panel.csv

# 4. Jalankan web dashboard
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
