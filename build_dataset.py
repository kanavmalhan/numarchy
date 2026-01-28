import pandas as pd
import numpy as np
import glob
import os
import time

# -----------------------------
# CONFIG
# -----------------------------
PRICE_FOLDER = "data/prices"
FIN_FOLDER = "data/financials"
OUTPUT_FILE = "data/ml_dataset.csv"
LOOKBACK_DAYS = [5, 20, 50]   # for returns / moving averages
FUTURE_DAYS = 126             # ~6 months for label
PAUSE_SECONDS = 1             # optional pause between stocks

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def merge_financials(prices, financials):
    """Merge financials with daily prices using latest report"""
    prices = prices.dropna(subset=['date'])
    financials = financials.dropna(subset=['report_date', 'symbol'])

    # Ensure datetime
    prices['date'] = pd.to_datetime(prices['date'])
    financials['report_date'] = pd.to_datetime(financials['report_date'])

    # Sort for merge_asof
    prices = prices.sort_values('date')
    financials = financials.sort_values('report_date')

    merged = pd.merge_asof(
        prices,
        financials,
        left_on='date',
        right_on='report_date',
        by='symbol',
        direction='backward'
    )
    return merged

def compute_technical_features(df):
    """Compute rolling returns, moving averages, volatility"""
    df = df.sort_values('date').copy()

    # Convert price columns to numeric
    price_cols = ['open','high','low','close','volume']
    for col in price_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    for n in LOOKBACK_DAYS:
        if 'close' in df.columns:
            df[f'return_{n}d'] = df['close'].pct_change(n)
            df[f'ma_{n}d'] = df['close'].rolling(n).mean()
            df[f'price_ma_ratio_{n}d'] = df['close'] / df[f'ma_{n}d']

    if 'close' in df.columns:
        df['volatility_20d'] = df['close'].pct_change().rolling(20).std()

    return df

def compute_labels(df, spy_prices):
    """Compute future return vs SPY and label"""
    df = df.sort_values('date').copy()

    # Stock future return
    df['future_return'] = df['close'].pct_change(FUTURE_DAYS, fill_method=None).shift(-FUTURE_DAYS)

    # SPY future return
    spy_future = spy_prices['close'].pct_change(FUTURE_DAYS, fill_method=None).shift(-FUTURE_DAYS)
    spy_future = pd.to_numeric(spy_future, errors='coerce')
    spy_future = spy_future.reindex(df['date']).ffill()

    # Align indexes
    spy_future = spy_future.reset_index(drop=True)
    df = df.reset_index(drop=True)

    df['label'] = (df['future_return'] > spy_future).astype(int)

    return df


# -----------------------------
# LOAD SPY PRICES
# -----------------------------
spy_prices = pd.read_csv(f"data/benchmark/SPY.csv", parse_dates=['date'])
for col in ['open','high','low','close','volume']:
    if col in spy_prices.columns:
        spy_prices[col] = pd.to_numeric(spy_prices[col], errors='coerce')
spy_prices = spy_prices.sort_values('date').set_index('date')

# -----------------------------
# PROCESS ALL STOCKS
# -----------------------------
all_files = glob.glob(f"{PRICE_FOLDER}/*.csv")
all_data = []

for file in all_files:
    symbol = os.path.basename(file).replace(".csv","")
    if symbol == "SPY":
        continue
    
    print(f"Processing {symbol}...")
    
    # Load prices and financials
    prices = pd.read_csv(file, parse_dates=['date'])
    fin_file = f"{FIN_FOLDER}/{symbol}.csv"
    if not os.path.exists(fin_file):
        print(f"Skipping {symbol}, no financials found")
        continue
    financials = pd.read_csv(fin_file, parse_dates=['report_date'])

    # Merge financials
    df = merge_financials(prices, financials)

    # Compute technical features
    df = compute_technical_features(df)

    # Compute labels
    df = compute_labels(df, spy_prices)

    # Drop rows with missing labels
    df = df.dropna(subset=['label'])

    all_data.append(df)
    time.sleep(PAUSE_SECONDS)

# -----------------------------
# COMBINE ALL STOCKS AND SAVE
# -----------------------------
if all_data:
    ml_df = pd.concat(all_data, ignore_index=True)
    ml_df.to_csv(OUTPUT_FILE, index=False)
    print(f"ML dataset saved to {OUTPUT_FILE}, shape: {ml_df.shape}")
else:
    print("No data to save.")
