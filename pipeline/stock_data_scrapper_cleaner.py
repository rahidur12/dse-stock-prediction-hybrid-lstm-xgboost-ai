# =============================
# stock_data_scrapper_cleaner.py
# =============================

import os
import pandas as pd
from bdshare import get_hist_data
from datetime import datetime
from dateutil.relativedelta import relativedelta

def download_and_fill_stock_data(symbol):
    # 1. Dynamic pathing: Locates the 'data' folder relative to this script
    current_script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_script_path))
    data_dir = os.path.join(project_root, "data")
    
    if not os.path.exists(data_dir): 
        os.makedirs(data_dir)

    # 2. Time Window (Fixed 2 years for thesis consistency)
    today = datetime.now()
    start_date = (today - relativedelta(years=2)).strftime('%Y-%m-%d')
    end_date = today.strftime('%Y-%m-%d')

    print(f"🚀 Fetching {symbol.upper()} stock history from {start_date} to {end_date}...")

    try:
        # Fetch data from DSE via bdshare
        df = get_hist_data(start_date, end_date, symbol.upper())

        if df is not None and not df.empty:
            df.index = pd.to_datetime(df.index)
            df = df.sort_index(ascending=True)

            # 3. Fill calendar gaps: Stock markets don't trade on weekends/holidays.
            # We reindex to 'D' (Daily) and forward-fill prices to keep the timeline continuous.
            calendar_days = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
            df = df.reindex(calendar_days).ffill()
            df = df.reset_index().rename(columns={'index': 'Date'})

            # 4. Standardized Filename: 
            # If symbol is 'GP', saves as 'gp_stock_data.csv'
            file_name = f"{symbol.lower()}_stock_data.csv"
            full_path = os.path.join(data_dir, file_name)

            df.to_csv(full_path, index=False)
            print(f"✅ Success! Saved {len(df)} rows to: {full_path}")
            return df
        else:
            print(f"❌ No data found for '{symbol}'. Check the symbol on DSE.")
    except Exception as e:
        print(f"⚠️ Connection or Data Error: {e}")

if __name__ == "__main__":
    # --- DYNAMIC INPUT SECTION ---
    user_symbol = input("Enter the DSE Stock Symbol (e.g., GP, BRACBANK, BEXIMCO): ").strip()
    if user_symbol:
        download_and_fill_stock_data(user_symbol)
    else:
        print("Invalid input. Please provide a valid symbol.")