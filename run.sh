uv sync

if [ $# -gt 0 ] && [ "$1" = '-db' ]; then
    echo "Creating database"
    uv run -m db.initialize_db
fi

PYTHONPATH=src uv run streamlit run src/app/main.py

