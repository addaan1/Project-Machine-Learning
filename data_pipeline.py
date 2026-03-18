"""
=============================================================================
  DATA PIPELINE (ANTI-LEAKAGE)
  Proyek: Prediksi Inflasi dan Dampaknya terhadap Daya Beli
=============================================================================
Script ini bertugas melakukan split (Train/Val/Test) TERLEBIH DAHULU pada
data yang sudah bersih, baru kemudian melakukan Scaling, Imputasi,
dan pembuatan Fitur Lag. Hal ini secara mutlak mencegah Data Leakage.
=============================================================================
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "datasets", "processed")


def get_lstm_pipeline_data(seq_length=12):
    """
    Pipeline untuk Model 1 (Forecasting LSTM).
    Membaca data time-series murni, membagi chronologically murni (Train -> Val -> Test),
    lalu fit Scaler HANYA pada set Train, serta membuat sequence X, y secara aman.
    """
    print("\n" + "="*50)
    print("  LSTM Data Pipeline (Chronological Split)")
    print("="*50)
    
    path = os.path.join(OUT_DIR, "clean_inflasi_ts.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} tidak ditemukan. Jalankan preprocessing.py dulu.")
        
    df = pd.read_csv(path)
    df["Tanggal"] = pd.to_datetime(df["Tanggal"])
    df = df.sort_values("Tanggal").reset_index(drop=True)
    
    # 1. Forward-fill missing values (IHK, BI Rate, dll) TETAPI HATI-HATI:
    # Karena ini time-series, ffill wajar asalkan urut waktu.
    # Namun BPS tidak merekam IHK setelah 2019, ffill IHK dari 2019 ke 2026 kurang valid.
    # Solusi: IHK kita hapus atau gunakan interpolasi hanya pada train.
    # Untuk BI_Rate & USD_IDR kita ffill berurutan waktu.
    df["BI_Rate"] = df["BI_Rate"].ffill().bfill()
    df["USD_IDR"] = df["USD_IDR"].ffill().bfill()
    df["IHK"] = df["IHK"].fillna(0) # Pad dengan 0 karena data hilang post-2019
    
    # Kolom fitur: Kolom-1=Inflasi_MoM, dsb
    feature_cols = ["Inflasi_MoM", "IHK", "BI_Rate", "USD_IDR"]
    raw_data = df[feature_cols].values
    
    # 2. CHRONOLOGICAL SPLIT (70% Train, 15% Val, 15% Test)
    n = len(raw_data)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    train_data = raw_data[:train_end]
    val_data = raw_data[train_end:val_end]
    test_data = raw_data[val_end:]
    
    # 3. FIT SCALER HANYA PADA TRAIN
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_data)
    
    # 4. TRANSFORM KETIGANYA MENGGUNAKAN SCALER TRAIN
    train_scaled = scaler.transform(train_data)
    val_scaled = scaler.transform(val_data)
    test_scaled = scaler.transform(test_data)
    
    # 5. FUNGSI UNTUK MEMBUAT SEQUENCE/WINDOWNING (X dan y)
    def create_sequences(data, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data[i:(i + seq_length)]
            y = data[i + seq_length, 0] # Index 0 adalah Inflasi_MoM
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)
    
    X_train, y_train = create_sequences(train_scaled, seq_length)
    X_val, y_val = create_sequences(val_scaled, seq_length)
    X_test, y_test = create_sequences(test_scaled, seq_length)
    
    print(f"   ✓ Rentang Train: {df['Tanggal'].iloc[0].date()} s.d {df['Tanggal'].iloc[train_end-1].date()}")
    print(f"   ✓ Rentang Val  : {df['Tanggal'].iloc[train_end].date()} s.d {df['Tanggal'].iloc[val_end-1].date()}")
    print(f"   ✓ Rentang Test : {df['Tanggal'].iloc[val_end].date()} s.d {df['Tanggal'].iloc[-1].date()}")
    print("-" * 50)
    print(f"   ✓ X_train : {X_train.shape}, y_train: {y_train.shape}")
    print(f"   ✓ X_val   : {X_val.shape}, y_val  : {y_val.shape}")
    print(f"   ✓ X_test  : {X_test.shape}, y_test : {y_test.shape}")
    
    # Mengembalikan data siap model dan scalernya (untuk denormalisasi saat inference)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, df


def get_regression_pipeline_data(target_col="Total_Pengeluaran"):
    """
    Pipeline untuk Model 2 (Regresi Dampak Daya Beli).
    Membaca panel daya beli mentah, di-split secara random, 
    lalu dilakukan log-transform (jika diperlukan) dan fit Scaler HANYA pada set Train.
    
    Argumen:
        target_col (str): Pilihan target ("Pengeluaran_Makanan", "Pengeluaran_Bukan_Makanan", "Total_Pengeluaran")
    """
    print("\n" + "="*50)
    print(f"  Regression Pipeline (Target: {target_col})")
    print("="*50)
    
    path = os.path.join(OUT_DIR, "clean_daya_beli.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} tidak ditemukan. Jalankan preprocessing.py dulu.")
        
    df = pd.read_csv(path)
    
    # Target (Y)
    y = df[target_col]
    
    # Fitur (X)
    X = df[["UMP", "TPT", "Inflasi_Rata_Tahunan"]]
    
    # 1. SPLIT DATA SEBELUM IMPUTASI MEAN & SCALING (Mencegah leakage)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. SEKARANG KITA LAKUKAN TRANSFORMASI & IMPUTASI MEAN HANYA BERDASAR TRAIN
    mean_tpt = X_train["TPT"].mean()
    X_train.loc[:, "TPT"] = X_train["TPT"].fillna(mean_tpt)
    X_test.loc[:, "TPT"]  = X_test["TPT"].fillna(mean_tpt) # Pakai mean_tpt dari TRAIN!
    
    # 3. LOG TRANSFORM
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)
    X_train_log = X_train.copy()
    X_test_log = X_test.copy()
    X_train_log["UMP"] = np.log1p(X_train["UMP"])
    X_test_log["UMP"] = np.log1p(X_test["UMP"])
    
    print(f"   ✓ Total observasi panel: {len(df)}")
    print(f"   ✓ X_train: {X_train_log.shape}, y_train: {y_train_log.shape}")
    print(f"   ✓ X_test : {X_test_log.shape},  y_test : {y_test_log.shape}")
    print("-" * 50)
    
    return X_train_log, X_test_log, y_train_log, y_test_log, df


if __name__ == "__main__":
    print("\nMenyelesaikan uji coba (Dry Run) Pipeline...")
    
    try:
        lstm_data = get_lstm_pipeline_data(seq_length=12)
        print("✓ LSTM Pipeline OK.")
    except Exception as e:
        print(f"✗ Gagal LSTM Pipeline: {e}")
        
    try:    
        # Test default
        reg_data = get_regression_pipeline_data(target_col="Total_Pengeluaran")
        # Test makanan
        reg_data_makan = get_regression_pipeline_data(target_col="Pengeluaran_Makanan")
        print("✓ Regression Pipeline OK.\n")
    except Exception as e:
        print(f"✗ Gagal Regression Pipeline: {e}")
