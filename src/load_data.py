import yfinance as yf
import pandas as pd

def load_price_data(ticker="AAPL", period="10y", interval="1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, threads=False)
    except Exception as e:
        print(f"YFinance download error: {e}")
        return pd.DataFrame()

    if df.empty:
        print("Downloaded DataFrame is empty. Possibly rate-limited.")
        return df

    df.reset_index(inplace=True)

    # If multi-index columns like ('Close', 'AAPL'), flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    # Keep only necessary columns
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df.dropna(inplace=True)
    return df
