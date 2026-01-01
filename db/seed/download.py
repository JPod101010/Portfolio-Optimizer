import yfinance as yf
import pandas as pd
import numpy as np
from db.seed.tickers import TICKERS
import pyarrow as pa
import pyarrow.parquet as pq


df_raw = yf.download(
    tickers=TICKERS,
    period='5y',
    interval='1d',
    auto_adjust=False,
    progress=True
)
df = df_raw.stack(level=1).rename_axis(['Date', 'Ticker']).reset_index()
df = df.sort_values(['Ticker', 'Date'])

df['RawChange'] = df.groupby('Ticker')['Close'].diff()
df['PctChange'] = df.groupby('Ticker')['Close'].pct_change()
df['LogChange'] = np.log1p(df['PctChange'].fillna(0))


#rename columns to match the sql schema
df_to_save = df.rename(columns={
    'Ticker': 'symbol',
    'Date': 'date_',
    'Open': 'open_price',
    'Close': 'close_price',
    'High': 'high_price',
    'Low': 'low_price',
    'RawChange': 'raw_diff',
    'PctChange': 'percent_diff',
    'LogChange': 'logpercent_diff',
    'Volume': 'volume'
})

df_to_save.to_parquet('db/seed/historical_prices.parquet', engine='pyarrow', index=False)