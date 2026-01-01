from sqlalchemy import create_engine
import os

DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://postgres:paswordik@db:5432/portfolio_db"
)

ENGINE = create_engine(DATABASE_URL)