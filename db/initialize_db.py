import os
import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path

DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://postgres:paswordik@localhost:5432/portfolio_db"
)


df = pd.read_parquet(Path(__file__).parent / 'seed' / 'historical_prices.parquet')
engine = create_engine(DATABASE_URL)

df.to_sql('prices', engine, if_exists='replace', index=False)