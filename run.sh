#!/bin/bash
set -e

echo "Waiting for database at db:5432..."
until curl -s http://db:5432 || [ $? -eq 52 ]; do
  echo "Postgres is unavailable - sleeping"
  sleep 1
done

if [ "$1" = '--db-init' ]; then
    echo "Initializing database..."
    DATABASE_URL=postgresql://postgres:paswordik@db:5432/portfolio_db \
    uv run -m db.initialize_db
fi

echo "Starting Streamlit..."
PYTHONPATH=src uv run streamlit run src/app/main.py --server.address 0.0.0.0