# Sistem Prediksi Inflasi & Analisis Dampak terhadap Daya Beli

> **Machine Learning SD-A1 — Universitas Airlangga**

## Ringkasan

Proyek ini mengembangkan sistem machine learning end-to-end untuk **memprediksi inflasi bulanan Indonesia** dan **menganalisis dampaknya terhadap daya beli masyarakat** di tingkat provinsi. Sistem terdiri dari dua model utama yang terintegrasi dalam dashboard web interaktif berbasis Django.

---

## Arsitektur Sistem

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Layer (23 Dataset)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │   Domestic   │  │ International│  │  World Bank API  │  │
│  │  (BPS, BI)   │  │(Yahoo, FRED) │  │  (Nasional)      │  │
│  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘  │
└─────────┼─────────────────┼───────────────────┼────────────┘
          │                 │                   │
          ▼                 ▼                   ▼
┌─────────────────────────────────────────────────────────────┐
│              Preprocessing Pipeline (Python)               │
│         preprocessing.py → clean_inflasi_ts.csv           │
│                         → clean_daya_beli.csv             │
└────────────────────────┬──────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         ▼                               ▼
┌─────────────────────┐      ┌─────────────────────────────┐
│ Model 1: Forecasting│      │ Model 2: Regresi Daya Beli  │
│      LSTM           │      │  Panel FE + XGBoost + Lasso │
│  (44 fitur, TS)     │      │  (18 fitur, Panel Data)     │
│  Test MAE: 0.18%    │      │  Test R²: 0.83 (XGBoost)    │
└──────────┬──────────┘      └─────────────┬───────────────┘
           │                               │
           └───────────────┬───────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│           Dashboard Django (Interactive Web App)            │
│   /forecasting/  →  Prediksi inflasi 1-2 bulan ke depan    │
│   /daya-beli/    →  Simulasi dampak kebijakan ekonomi     │
└─────────────────────────────────────────────────────────────┘
```

---

## Model 1 — Forecasting Inflasi (LSTM)

**Prediksi inflasi bulanan 1-2 bulan ke depan** menggunakan 44 fitur time-series:

| Aspek | Spesifikasi |
|-------|-------------|
| Arsitektur | LSTM 2-layer, 128 hidden units, LayerNorm, Dropout 0.3 |
| Window | 12 bulan |
| Fitur | 44 (1 target + 43 eksogenous: domestik, internasional, komoditas) |
| Komoditas | 26 komoditas World Bank CMO (minyak, batu bara, CPO, pangan, logam) |
| Split | Chronological: 80% Train, 20% Validation |
| Scaler | MinMaxScaler (fitur & target terpisah) |
| Optimizer | AdamW + ReduceLROnPlateau + Early Stopping |
| Metrik | MAE: 0.18% (test) |

**Data Sources**: BPS (IHK, inflasi komponen), Bank Indonesia (BI Rate), Yahoo Finance (Brent, DXY, Gold, CPO), FRED (Fed Funds Rate), FAO (Food Price Index), World Bank CMO (26 komoditas), Geopolitical Risk Index.

---

## Model 2 — Analisis Daya Beli (Panel Data + ML)

**Mengukur pengaruh makroekonomi terhadap daya beli per kapita** di 38 provinsi Indonesia (2021–2025).

### Pendekatan

| Model | Metode | Tujuan | Test R² |
|-------|--------|--------|---------|
| Baseline OLS | 3 fitur (Inflasi, UMP, Tahun) | Interpretasi sederhana | 0.18 |
| **Panel FE Macro** | Fixed Effects + 4 fitur makro | **Model interpretatif utama** | 0.43 |
| Panel FE Full | Fixed Effects + 18 fitur | Eksplorasi lengkap | overfit |
| Lasso | L1 regularization | Feature selection otomatis | 0.68 |
| Random Forest | Tree ensemble | Non-linear relationships | 0.69 |
| **XGBoost** | Gradient boosting | **Model prediktif terbaik** | **0.83** |

### Feature Engineering

- **Real_UMP** = UMP / (1 + Inflasi) → daya beli upah riil
- **YoY Growth** → pertumbuhan tahunan per provinsi
- **Interaction** → UMP × PDRB, Inflasi × TPT
- **Log transforms** → untuk distribusi skewed

### Temuan Ekonomi (Panel FE Macro)

| Variabel | Efek | Signifikansi | Elastisitas |
|----------|------|--------------|-------------|
| **TPT** (Pengangguran) | Negatif | ⭐⭐⭐ (t=-6.97) | -0.30% |
| **PDRB** | Positif | ⭐⭐⭐ (t=4.25) | +0.26% |
| **Real_UMP** | Positif | ⭐⭐⭐ (t=4.18) | +0.51% |
| **Inflasi** | Positif kecil | ⭐⭐ (t=2.98) | +0.05% |

### Simulasi Counterfactual

| Skenario Kebijakan | Dampak Daya Beli |
|--------------------|------------------|
| UMP naik 10% | **+4.7%** |
| Inflasi +5% & UMP +10% | **+4.9%** |
| PDRB turun 20% | -5.3% |
| Resesi (TPT ×3) | **-56.0%** |

---

## Dataset

Total **23 dataset** dari 11 sumber resmi:

### Model 1 (Forecasting) — 15 Dataset, 44 Fitur

| Kategori | Dataset | Sumber |
|----------|---------|--------|
| Domestik | IHK, Inflasi M-to-M, BI Rate, USD/IDR, Inflasi Komponen, WTI | BPS, BI, Investing.com, IndexMundi |
| Internasional | Brent, DXY, Fed Funds, Gold, CPO, GPR, FAO Food Price | Yahoo Finance, FRED, FAO, policyuncertainty.com |
| Komoditas | 26 komoditas (minyak, batu bara, CPO, pangan, logam) | World Bank Commodity Markets Outlook |

### Model 2 (Daya Beli) — 8 Dataset, 18 Fitur

| Kategori | Dataset | Sumber |
|----------|---------|--------|
| Makro Provinsi | UMP, PDRB, TPT, Pengeluaran per Kapita | BPS, Open Data Jabar |
| Sosial | IPM, Jumlah Penduduk, Distribusi Penduduk | BPS |
| Indikator Nasional | GDP per Capita PPP, Unemployment, Poverty, Inflasi | World Bank API |

---

## Teknologi

| Layer | Stack |
|-------|-------|
| **Data Pipeline** | Python, Pandas, NumPy |
| **Modeling** | PyTorch (LSTM), Scikit-learn (Ridge, Lasso), XGBoost, statsmodels, linearmodels |
| **Dashboard** | Django, Chart.js |
| **Visualization** | Matplotlib, Seaborn |
| **Version Control** | Git |

---

## Struktur Repository

```
├── datasets/               # Raw & processed data
│   ├── international/        # Yahoo, FRED, FAO, World Bank CMO
│   ├── domestic_baru/       # BPS per-provinsi
│   └── processed/            # clean_inflasi_ts.csv, clean_daya_beli.csv
├── notebooks/               
│   ├── forecasting_inflasi_models.ipynb
│   └── analisis_daya_beli_regresi.ipynb   # Panel FE + XGBoost + Counterfactual
├── dashboard/               # Django web app
│   └── predictions/
│       └── views.py          # Model loading & API endpoints
├── preprocessing.py         # ETL pipeline
├── download_international.py # Auto-download Yahoo, FRED, FAO
├── download_domestic.py      # Auto-download World Bank API
├── save_lstm_model.py      # Training pipeline LSTM
├── save_ridge_model.py     # Training pipeline Ridge
├── models/                  # Serialized models
│   ├── lstm_inflasi.pt
│   ├── best_daya_beli_ridge.pkl
│   └── best_daya_beli_xgboost.pkl
└── requirements.txt
```

---

## Hasil & Metrik

| Model | Train R² | Test R² | Test MAE |
|-------|----------|---------|----------|
| LSTM Inflasi | — | — | **0.18%** |
| XGBoost Daya Beli | 1.000 | **0.834** | Rp 119,224 |
| Random Forest Daya Beli | 0.859 | 0.693 | Rp 142,937 |
| Lasso Daya Beli | 0.848 | 0.679 | Rp 154,513 |
| Panel FE Macro | 0.986 | 0.427 | Rp 126,334 |

---

## Dashboard

| Endpoint | Fungsi |
|----------|--------|
| `/` | Overview inflasi & daya beli nasional |
| `/forecasting/` | Prediksi inflasi 1-2 bulan ke depan dengan chart interaktif |
| `/daya-beli/` | Simulasi slider: UMP, inflasi, TPT, PDRB → prediksi daya beli |

---

## Tim Pengembang

| Nama | NIM |
|------|-----|
| Muhammad Rajif Al Farikhi | 162112133008 |
| Sahrul Adicandra Effendy | 164231013 |
| Semaya David Petroes Putra | 164231048 |
| Adrina Firda Marwah | 164231087 |
| Okan Athallah Maredith | 164231088 |

---

## Referensi Data

- [Badan Pusat Statistik (BPS)](https://www.bps.go.id)
- [Bank Indonesia](https://www.bi.go.id)
- [World Bank Commodity Markets Outlook](https://www.worldbank.org/en/research/commodity-markets)
- [Yahoo Finance](https://finance.yahoo.com)
- [FRED](https://fred.stlouisfed.org)
- [FAO Food Price Index](https://www.fao.org/worldfoodsituation/foodpricesindex/en/)
- [Geopolitical Risk Index](https://www.policyuncertainty.com/gpr.html)
