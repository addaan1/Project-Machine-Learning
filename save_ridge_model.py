"""
============================================================================
  RETRAIN RIDGE MODEL (Model 2: Regresi Daya Beli)
  Proyek: Prediksi Inflasi dan Dampaknya terhadap Daya Beli
============================================================================
Train ulang model Ridge Regression dengan fitur terbaru dari clean_daya_beli.csv.
Output: models/best_daya_beli_ridge.pkl (pipeline + metadata)
============================================================================
"""

import os
import pickle
import warnings
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "datasets", "processed", "clean_daya_beli.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def main():
    print("=" * 65)
    print("  RETRAIN RIDGE MODEL (Model 2: Regresi Daya Beli)")
    print("=" * 65)

    if not os.path.exists(DATA_PATH):
        print(f"  [GAGAL] {DATA_PATH} tidak ditemukan. Jalankan preprocessing.py dulu.")
        return

    df = pd.read_csv(DATA_PATH)
    print(f"\n  Dataset: {df.shape[0]} baris × {df.shape[1]} kolom")
    print(f"  Kolom: {list(df.columns)}")
    
    # --- Feature Engineering: Real_UMP ---
    if 'UMP' in df.columns and 'Inflasi_Rata_Tahunan' in df.columns:
        df['Real_UMP'] = df['UMP'] / (1 + df['Inflasi_Rata_Tahunan'])
    else:
        print("  [ERROR] UMP atau Inflasi_Rata_Tahunan tidak ada, tidak bisa hitung Real_UMP")
        return
    
    # --- Drop kolom redundan (PDRB_HargaBerlaku, TPAK, Pct_Penduduk_Miskin) ---
    # PDRB_HargaBerlaku vs PDRB_HargaKonstan: pilih Konstan (sudah di-inflate)
    # TPAK dan Pct_Penduduk_Miskin: missing values tinggi
    cols_to_drop = ['UMP', 'PDRB_HargaBerlaku', 'TPAK', 'Pct_Penduduk_Miskin',
                    'Pengeluaran_Makanan', 'Pengeluaran_Bukan_Makanan']
    df_clean = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # --- Identifikasi fitur numerik vs kategorikal ---
    target_col = 'Total_Pengeluaran'
    cat_features = ['Provinsi']
    
    # Pilih fitur numerik: Semua kecuali identitas & target
    num_features = [c for c in df_clean.columns 
                    if c not in [target_col] + cat_features + ['Tahun']]
    
    # Hanya gunakan fitur dengan data >= 50% non-null
    num_features = [f for f in num_features 
                    if df_clean[f].notna().sum() >= len(df_clean) * 0.5]
    
    print(f"\n  Fitur numerik: {len(num_features)}")
    for f in num_features:
        null_pct = df_clean[f].isna().sum() / len(df_clean) * 100
        print(f"    - {f}: {100-null_pct:.0f}% non-null")
    
    # Drop rows dengan NaN pada fitur yang akan dipakai
    df_model = df_clean[['Tahun'] + num_features + cat_features + [target_col]].dropna()
    print(f"\n  Setelah drop NaN: {df_model.shape[0]} baris")
    
    # --- Train-Test Split (chronological) ---
    train_mask = df_model['Tahun'] <= 2023
    test_mask = df_model['Tahun'] >= 2024
    train_df = df_model[train_mask]
    test_df = df_model[test_mask]
    
    X_train = train_df[num_features + cat_features]
    y_train = train_df[target_col]
    X_test = test_df[num_features + cat_features]
    y_test = test_df[target_col]
    
    print(f"\n  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    
    # --- Pipeline Preprocessing ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
        ]
    )
    
    # --- Hyperparameter Tuning dengan GridSearchCV ---
    print("\n  Tuning hyperparameter dengan GridSearchCV...")
    param_grid = {
        'regressor__alpha': [0.01, 0.1, 1.0, 5.0, 10.0, 50.0],
    }
    
    base_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', Ridge())
    ])
    
    grid = GridSearchCV(
        base_pipeline,
        param_grid,
        cv=3,
        scoring='r2',
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_train, y_train)
    
    best_pipeline = grid.best_estimator_
    print(f"  Best alpha: {grid.best_params_['regressor__alpha']}")
    
    # --- Evaluasi ---
    y_pred_train = best_pipeline.predict(X_train)
    y_pred_test = best_pipeline.predict(X_test)
    
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    print(f"\n  --- Evaluasi Model (Best Ridge) ---")
    print(f"  Train R² : {train_r2:.4f}")
    print(f"  Test R²  : {test_r2:.4f}")
    print(f"  Test MAE : Rp {test_mae:,.2f}")
    print(f"  Test RMSE: Rp {test_rmse:,.2f}")
    
    # --- Simpan Model ---
    model_bundle = {
        'pipeline': best_pipeline,
        'num_features': num_features,
        'cat_features': cat_features,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'best_alpha': grid.best_params_['regressor__alpha'],
    }
    
    out_path = os.path.join(MODEL_DIR, 'best_daya_beli_ridge.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(model_bundle, f)
    
    print(f"\n  ✓ Model disimpan → {out_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
