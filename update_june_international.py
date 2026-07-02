import os
import pandas as pd
import yfinance as yf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INT_DIR = os.path.join(BASE_DIR, "datasets", "international")

def update_csv_monthly(filename, date_col, val_col, new_date, new_val):
    path = os.path.join(INT_DIR, filename)
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    df = pd.read_csv(path)
    if new_date in df[date_col].values:
        print(f"[{filename}] {new_date} already present. Updating value to {new_val}")
        df.loc[df[date_col] == new_date, val_col] = new_val
    else:
        print(f"[{filename}] Appending {new_date} -> {new_val}")
        new_row = pd.DataFrame([{date_col: new_date, val_col: new_val}])
        df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(path, index=False)

def update_usd_idr_daily():
    path = os.path.join(INT_DIR, "usd_idr_2026.csv")
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    df_existing = pd.read_csv(path)
    print("Downloading June 2026 IDR=X from Yahoo Finance...")
    df_new = yf.download("IDR=X", start="2026-06-01", end="2026-07-01")
    if df_new.empty:
        print("No new data downloaded for IDR=X.")
        return
    
    # Flatten multi-index columns if present
    if isinstance(df_new.columns, pd.MultiIndex):
        df_new.columns = [col[0] for col in df_new.columns]
    
    df_new = df_new.reset_index()
    # Format date as YYYY-MM-DD string
    df_new["Date"] = pd.to_datetime(df_new["Date"]).dt.strftime("%Y-%m-%d")
    
    # Ensure standard columns exist
    for col in ["Adj Close", "Close", "High", "Low", "Open", "Volume"]:
        if col not in df_new.columns and col in df_existing.columns:
            df_new[col] = df_new["Close"] if "Close" in df_new.columns else 0
            
    df_new = df_new[["Date", "Adj Close", "Close", "High", "Low", "Open", "Volume"]]
    
    # Merge avoiding duplicate dates
    existing_dates = set(df_existing["Date"].astype(str).values)
    to_append = df_new[~df_new["Date"].isin(existing_dates)]
    
    if not to_append.empty:
        print(f"Appending {len(to_append)} rows to usd_idr_2026.csv")
        df_combined = pd.concat([df_existing, to_append], ignore_index=True)
        df_combined = df_combined.sort_values("Date").reset_index(drop=True)
        df_combined.to_csv(path, index=False)
    else:
        print("No new dates to append for usd_idr_2026.csv.")

def main():
    print("=== Updating International Datasets for June 2026 ===")
    update_csv_monthly("crude_oil_brent.csv", "Tanggal", "Brent_USD", "2026-06-01", 72.919998)
    update_csv_monthly("dxy_dollar_index.csv", "Tanggal", "DXY", "2026-06-01", 101.190002)
    update_csv_monthly("gold_price.csv", "Tanggal", "Gold_USD", "2026-06-01", 4022.899902)
    update_csv_monthly("fed_funds_rate.csv", "Tanggal", "FedRate_Pct", "2026-06-01", 3.63)
    update_csv_monthly("cpo_price.csv", "Tanggal", "CPO_USD", "2026-06-01", 1151.4250)
    update_csv_monthly("wti_apr_may_2026.csv", "Tanggal", "Harga", "2026-06-01", 69.500000)
    
    update_usd_idr_daily()
    print("=== Update Complete ===")

if __name__ == "__main__":
    main()
