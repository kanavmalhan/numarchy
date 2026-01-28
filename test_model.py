import pandas as pd
import numpy as np
import joblib
import glob
import os


MODEL_PATH = "models/xgb_model.pkl"
PRICE_FOLDER = "data/prices"
FIN_FOLDER = "data/financials"
LOOKBACK_DAYS = [5, 20, 50]
MAX_MISSING_RATIO = 0.8

# Load the model first to get training features
print("Loading trained model...")
model = joblib.load(MODEL_PATH)
print("Model loaded from", MODEL_PATH)

# Get feature names directly from the model
training_features = model.get_booster().feature_names
print(f"Training features: {len(training_features)}")

DROP_COLS = [
    'date',
    'symbol',
    'report_date',
    'label',
    'future_return'
]

# ==========================================
# Now load recent data for each stock
# ==========================================
print("\nLoading recent data for all stocks...")

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

all_recent_data = []
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
        
        # Ensure datetime columns and drop nulls
        prices['date'] = pd.to_datetime(prices['date'], errors='coerce')
        prices = prices.dropna(subset=['date'])
        
        if 'report_date' in financials.columns:
            financials['report_date'] = pd.to_datetime(financials['report_date'], errors='coerce')
            financials = financials.dropna(subset=['report_date'])
        
        if len(prices) == 0:
            continue
        
        # Merge using same logic as build_dataset.py
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
        
        # Get most recent row (drop nulls for key columns)
        recent = merged.iloc[[-1]].copy()
        
        # Only include if we have close price
        if 'close' in recent.columns and not recent['close'].isna().any():
            all_recent_data.append(recent)
            all_symbols.append(symbol)
            print(f"✓ {symbol}")
    except Exception as e:
        print(f"✗ {symbol}: {e}")
        continue

if not all_recent_data:
    print("No data loaded!")
else:
    df_recent = pd.concat(all_recent_data, ignore_index=True)
    print(f"Loaded recent data for {len(all_symbols)} stocks")
    
    # Preprocess
    X = df_recent.drop(columns=[c for c in DROP_COLS if c in df_recent.columns])
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.select_dtypes(include=[np.number])
    X = X.fillna(X.median(numeric_only=True))
    
    # Ensure X has the same columns as training
    for col in training_features:
        if col not in X.columns:
            X[col] = 0
    
    X = X[training_features]
    
    print(f"Prediction data shape: {X.shape}")
    
    print("Loading trained model...")
    model = joblib.load(MODEL_PATH)
    print("Model loaded from", MODEL_PATH)
    
    probs = model.predict_proba(X)[:, 1]
    
    # Get top 20 predictions
    results = pd.DataFrame({
        'symbol': all_symbols,
        'probability': probs
    })
    results = results.sort_values('probability', ascending=False).head(20)
    
    print("\n" + "="*50)
    print("TOP 20 PREDICTIONS")
    print("="*50)
    for idx, row in results.iterrows():
        print(f"{row['symbol']}: {row['probability']:.4f}")

print("\n" + "="*50)
print("TOP 20 PREDICTIONS")
print("="*50)
for idx, row in results.iterrows():
    print(f"{row['symbol']}: {row['probability']:.4f}")