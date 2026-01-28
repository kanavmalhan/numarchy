import yfinance as yf
import pandas as pd
import os

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
def get_sp500_symbols():
    df = pd.read_csv('constituents.csv')
    sp500_symbols = df['Symbol'].tolist()
    return sp500_symbols
def save_sp500():
    sp500_symbols = get_sp500_symbols()
    for i in range(len(sp500_symbols)):
        symbol = sp500_symbols[i]
        print(f"Fetching historical prices for {symbol} ({i+1}/{len(sp500_symbols)})...")
        data = yf.download(symbol, start=START_DATE, progress=False)
        if not data.empty:
            save_prices(data, symbol, f"{DATA_FOLDER}/prices")
        else:
            print(f"No data found for {symbol}. Skipping.")
COLUMN_MAPPING = {
    'Total Revenue': 'revenue',
    'Net Income': 'net_income',
    'Earnings Per Share (EPS)': 'eps',
    'Total Assets': 'total_assets',
    'Total Liab': 'total_liabilities',
    'Total Cash From Operating Activities': 'free_cash_flow'
}

def fetch_financials(symbol):
    print(f"Fetching financials for {symbol}...")
    ticker = yf.Ticker(symbol)
    
    # Get sector info
    sector = ticker.info.get('sector', None)

    # Fetch quarterly data
    try:
        income = ticker.quarterly_financials.T
        balance = ticker.quarterly_balance_sheet.T
        cashflow = ticker.quarterly_cashflow.T
    except Exception as e:
        print(f"Failed to fetch {symbol}: {e}")
        return None

    if income.empty:
        print(f"No financials for {symbol}")
        return None

    # Merge all three
    df = income.join(balance, how='outer', rsuffix='_bal')
    df = df.join(cashflow, how='outer', rsuffix='_cf')

    # Keep all numeric columns
    df = df.select_dtypes(include='number')

    # Add reference columns
    df['symbol'] = symbol
    df['report_date'] = df.index
    df['sector'] = sector

    # Reorder columns: symbol, report_date, all numeric financials, sector
    cols = ['symbol', 'report_date'] + [c for c in df.columns if c not in ['symbol','report_date','sector']] + ['sector']
    df = df[cols]

    # Convert report_date to datetime
    df['report_date'] = pd.to_datetime(df['report_date'])

    return df

def save_financials_sp500():
    sp500_symbols = get_sp500_symbols()
    for i in range(len(sp500_symbols)):
        symbol = sp500_symbols[i]
        financials = fetch_financials(symbol)
        if financials is not None:
            financials.to_csv(f"{DATA_FOLDER}/financials/{symbol}.csv", index=False)
            print(f"Saved financials for {symbol}: {len(financials)} rows")
        else:
            print(f"No financials data for {symbol}. Skipping.")

    else:
        print("No financials data to save.")

save_financials_sp500()
if(update_benchmark):
    print(f"Fetching historical prices for {BENCHMARK}...")
    spy = yf.download(BENCHMARK, start=START_DATE, progress=False)
    save_prices(spy, BENCHMARK, f"{DATA_FOLDER}/benchmark")
print("Data population complete.")