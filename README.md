## How to run

```bash
git clone https://github.com/JPod101010/Portfolio-Optimizer.git
cd Portfolio-Optimizer
docker compose up --build
```

To run locally (unrecommended):
```bash
chmod u+x run.sh
./run.sh --db-init
```

- Argument `--db-init` initializes database
- This also requires the change of `src/app/database_engine.py` to localhost

## Dear reader
- The project was build purely in python (with a little SQL)
- The aim of this project is to give the user some visuals and
understanding of quantitave basics
- Frontend was done in **streamlit** because i cannot code frontend in real
frontend tools
- If anyone is interest in explanation of the math concepts behind the visuals and computation
go see **docs/**
- The project is still WIP but has already some cool modules!