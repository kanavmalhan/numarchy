import yfinance as yf
import pandas as pd
import os

SYMBOL = "MU"
BENCHMARK = "SPY"
START_DATE = "2010-01-01"
DATA_FOLDER = "data"

update_benchmark = False

os.makedirs(f"{DATA_FOLDER}/prices", exist_ok=True)
os.makedirs(f"{DATA_FOLDER}/benchmark", exist_ok=True)

def save_prices(df, symbol, folder):
    df = df.reset_index()
    df = df[['Date', 'Close', 'Volume']]
    df.rename(columns={
        'Date': 'date',
        'Close': 'close',
        'Volume': 'volume'
    }, inplace=True)
    
    df['symbol'] = symbol

    df = df[['symbol', 'date', 'close', 'volume']]
    
    df = df.sort_values('date')
    
    df.to_csv(f"{folder}/{symbol}.csv", index=False)
    print(f"Saved {symbol} prices: {len(df)} rows")
def save_sp500():
    df = pd.read_csv('constituents.csv')
    sp500_symbols = df['Symbol'].tolist()
    print(sp500_symbols)
    for i in range(len(sp500_symbols)):
        symbol = sp500_symbols[i]
        print(f"Fetching historical prices for {symbol} ({i+1}/{len(sp500_symbols)})...")
        data = yf.download(symbol, start=START_DATE, progress=False)
        if not data.empty:
            save_prices(data, symbol, f"{DATA_FOLDER}/prices")
        else:
            print(f"No data found for {symbol}. Skipping.")

save_sp500()
if(update_benchmark):
    print(f"Fetching historical prices for {BENCHMARK}...")
    spy = yf.download(BENCHMARK, start=START_DATE, progress=False)
    save_prices(spy, BENCHMARK, f"{DATA_FOLDER}/benchmark")
print("Data population complete.")