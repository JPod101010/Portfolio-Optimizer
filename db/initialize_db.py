import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path


df = pd.read_parquet(Path(__file__).parent / 'seed' / 'historical_prices.parquet')

engine = create_engine('postgresql://postgres:paswordik@localhost:5432/portfolio_db')

df.to_sql('prices', engine, if_exists='replace', index=False)