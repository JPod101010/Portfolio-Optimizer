import yfinance as yf
from pathlib import Path
import pandas as pd, numpy as np

PATH_TO_DATA = Path(__file__).parent.parent.parent / "data" / "portfolio"

def compute_returs(df : pd.DataFrame) -> pd.DataFrame:
    df['RawChange'] = df['Close'].diff()
    df['PctChange'] = df['Close'].pct_change()
    df['LogChange'] = np.log1p(df['PctChange'])

    return df

def get_timeseries_by_ticker(ticker : str):

    PATH_TO_TICKER = f"{PATH_TO_DATA}/{ticker}.csv"
    try:
        df = pd.read_csv(PATH_TO_TICKER, index_col=0, parse_dates=True)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.iloc[:, 0])
            df.drop(df.columns[0], axis=1, inplace=True)
    except(FileNotFoundError):
        df_raw = yf.download(
            ticker,
            period='5y',
            interval='1d',
            auto_adjust=False,
            progress=False
        )
        
        if isinstance(df_raw.columns, pd.MultiIndex):
            df_raw.columns = df_raw.columns.droplevel(1)

        df = df_raw
        df.to_csv(PATH_TO_TICKER)

    return df
