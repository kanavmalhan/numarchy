import pandas as pd
import numpy as np
import joblib
import glob
import os
from datetime import datetime, timedelta


MODEL_PATH = "models/xgb_model.pkl"
PRICE_FOLDER = "data/prices"
FIN_FOLDER = "data/financials"
LOOKBACK_DAYS = [5, 20, 50]
MAX_MISSING_RATIO = 0.8

print("Loading trained model...")
model = joblib.load(MODEL_PATH)
print("Model loaded from", MODEL_PATH)

training_features = model.get_booster().feature_names
print(f"Training features: {len(training_features)}")

DROP_COLS = [
    'date',
    'symbol',
    'report_date',
    'label',
    'future_return'
]

backtest_date = datetime.now() - timedelta(days=365)
print(f"\nBacktest date: {backtest_date.strftime('%Y-%m-%d')}")
print("Loading historical data for all stocks...")

def compute_technical_features(df):
    """Compute rolling returns, moving averages, volatility"""
    df = df.sort_values('date').copy()
    for col in ['open','high','low','close','volume']:
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

all_historical_data = []
all_symbols = []

price_files = glob.glob(f"{PRICE_FOLDER}/*.csv")
for file in sorted(price_files):
    symbol = os.path.basename(file).replace(".csv", "")
    if symbol == "SPY":
        continue
    
    try:
        prices = pd.read_csv(file)
        fin_file = f"{FIN_FOLDER}/{symbol}.csv"
        
        if not os.path.exists(fin_file):
            continue
        
        financials = pd.read_csv(fin_file)
        
        prices['date'] = pd.to_datetime(prices['date'], errors='coerce')
        prices = prices.dropna(subset=['date'])
        
        if 'report_date' in financials.columns:
            financials['report_date'] = pd.to_datetime(financials['report_date'], errors='coerce')
            financials = financials.dropna(subset=['report_date'])
        
        if len(prices) == 0:
            continue
        
        prices = prices[prices['date'] <= backtest_date]
        
        if len(prices) == 0:
            continue
        
        prices = prices.sort_values('date')
        if 'report_date' in financials.columns and len(financials) > 0:
            financials = financials.sort_values('report_date')
            merged = pd.merge_asof(
                prices, financials,
                left_on='date', right_on='report_date',
                by='symbol' if 'symbol' in financials.columns else None,
                direction='backward'
            )
        else:
            merged = prices
        
        # Add technical features
        merged = compute_technical_features(merged)
        
        # Get row closest to backtest date
        merged['date_diff'] = (merged['date'] - backtest_date).dt.days.abs()
        historical = merged.loc[merged['date_diff'].idxmin():merged['date_diff'].idxmin()].copy()
        
        # Only include if we have close price
        if 'close' in historical.columns and not historical['close'].isna().any():
            all_historical_data.append(historical)
            all_symbols.append(symbol)
            actual_date = historical['date'].iloc[0].strftime('%Y-%m-%d')
            print(f"✓ {symbol} (data from {actual_date})")
    except Exception as e:
        print(f"✗ {symbol}: {e}")
        continue

if not all_historical_data:
    print("No data loaded!")
else:
    df_historical = pd.concat(all_historical_data, ignore_index=True)
    print(f"\nLoaded historical data for {len(all_symbols)} stocks")
    
    # Preprocess
    X = df_historical.drop(columns=[c for c in DROP_COLS if c in df_historical.columns])
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.select_dtypes(include=[np.number])
    X = X.fillna(X.median(numeric_only=True))
    
    # Ensure X has the same columns as training
    for col in training_features:
        if col not in X.columns:
            X[col] = 0
    
    X = X[training_features]
    
    print(f"Prediction data shape: {X.shape}")
    
    probs = model.predict_proba(X)[:, 1]
    
    # Get top 20 predictions
    results = pd.DataFrame({
        'symbol': all_symbols,
        'probability': probs
    })
    results = results.sort_values('probability', ascending=False).head(20)
    
    print("\n" + "="*50)
    print(f"TOP 20 PREDICTIONS (as of {backtest_date.strftime('%Y-%m-%d')})")
    print("="*50)
    for idx, row in results.iterrows():
        print(f"{row['symbol']}: {row['probability']:.4f}")
