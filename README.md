# Used-Car Price MLOps (Pakistan)

Predict a used car's price from a CSV, and run it like a product:
**clean data → train + track → register → serve (API + dashboard) → monitor drift → retrain → CI → Docker.**

> Batches are SIMULATED from one CSV (batch 3 = newer cars, prices x1.25) to demo drift. Say this in class.

## Project map
```
data/raw/cars_raw.csv     original data         src/    pipeline code
data/batches/             4 simulated batches   api/    FastAPI service
reports/                  drift + EDA outputs   dashboard/  Streamlit app
tests/                    13 pytest tests       docker/, .github/workflows/ci.yml
```

---
## PART A: Run it locally, end to end

You will use **4 terminals**. Keep each one open. In every terminal, first run:
```
cd ~/Downloads/car-price-mlops          # or wherever you unzipped it
conda activate carprice
export MLFLOW_TRACKING_URI=http://127.0.0.1:5001
```
(`export` is needed in every new terminal on Mac/Linux. On Windows use `set MLFLOW_TRACKING_URI=http://127.0.0.1:5001`.)

### Step 0. One-time setup (Python 3.11)
Use Python **3.11 or 3.12**. Python 3.13 fails (see Problem 1).
```
conda create -n carprice python=3.11 -y
conda activate carprice
pip install -r requirements.txt
python -c "import mlflow, xgboost, sklearn; print('ok')"     # must print: ok
```
No conda? `python3.11 -m venv venv`, then `source venv/bin/activate`.

### Step 1. Terminal 1: start MLflow (tracking + registry)
```
mlflow server --host 127.0.0.1 --port 5001 --serve-artifacts
```
Open http://127.0.0.1:5001. We use port **5001** because 5000 is often taken on Mac (Problem 2).

### Step 2. Terminal 2: data, training, registry
```
python -m src.make_batches     # clean raw CSV, split into 4 batches, write reports
python -m src.train            # 4 runs: ridge + 3 xgboost, logged to MLflow
python -m src.register         # best run -> alias "production"
```
Expected last line: `production -> car-price v1 (xgb_depth12, mae ...)`.
Check MLflow: experiment `car-price-regression` has 4 runs.

### Step 3. Terminal 3: API
```
uvicorn api.main:app --port 8000
```
Check http://127.0.0.1:8000/health shows `"model_loaded": true`.
Docs and test form: http://127.0.0.1:8000/docs

### Step 4. Terminal 4: dashboard
```
streamlit run dashboard/app.py
```
Open http://localhost:8501 and estimate a price.

### Step 5. Tests and quality gate (any terminal)
```
pytest -q                      # 13 passed
python -m src.quality_gate     # model must be clearly better than guessing the average
```

### Step 6. Monitor and retrain (drift demo)
```
python -m src.drift 3          # Evidently report -> reports/drift.html (open in browser)
python -m src.retrain          # retrain on new data; promote ONLY if MAE is better; reload API
```
Expected: `PROMOTED v2`. Then `/health` shows version 2 and the dashboard sidebar updates.
Why MAE: RMSE is thrown off by a few very expensive cars.
Optional: `python flows/weekly_pipeline.py` runs the same as a Prefect flow.

---
## PART B: Docker (do this after Part A works)
1. Start **Docker Desktop** and wait until it says running.
2. Stop the local MLflow/API/Streamlit (Ctrl+C), then:
```
docker compose up --build
```
MLflow :5000, API :8000, dashboard :8501. Containers talk by service name, not localhost.
3. Train against the container's MLflow (a new terminal):
```
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
python -m src.train && python -m src.register
curl -X POST http://127.0.0.1:8000/reload
```
Stop: `docker compose down`.

---
## PART C: CI (GitHub Actions)
Push to GitHub. `.github/workflows/ci.yml` runs pytest, the quality gate, and the 3 Docker builds on every push.

---
## 5-minute class demo
1. MLflow UI: compare the 4 runs.  2. Dashboard: estimate a price.
3. `python -m src.drift 3`: show the drift report.  4. `python -m src.retrain`: PROMOTED, v2 live.
5. Show the green CI run.

---
## Problems we hit and fixes

| # | Problem | Cause | Fix |
|---|---|---|---|
| 1 | `pip install` fails while building numpy (`ptrdiff_t`, `FAILED ... string_ufuncs.cpp`) | Python 3.13: no prebuilt numpy 2.0.2, pip compiles it and fails | Use Python 3.11 (Step 0) |
| 2 | MLflow: `Address already in use` port 5000 | Mac AirPlay Receiver, an old MLflow, or Docker uses 5000 | Run MLflow on 5001 and export `MLFLOW_TRACKING_URI` |
| 3 | `ImportError: cannot import name 'add_features'` | File not saved (empty `features.py`) | Save all files (Cmd+S); check with `wc -c src/features.py` |
| 4 | Pandera validation errors (`engine_cc`, year, engine range) | Raw data has strings and bad rows | Run `python -m src.make_batches` (cleans before splitting) |
| 5 | Command fails after pasting | One command split over two lines | Paste each command on a single line |
| 6 | Docker: cannot connect to daemon | Docker Desktop not running | Open Docker Desktop, wait until running |
| 7 | Docker build: `reports` not found | Folder missing | `mkdir -p reports && touch reports/.keep` |
| 8 | mlflow container crashes: `FallbackAsyncAdaptedQueuePool` | SQLAlchemy 2.1 breaks MLflow 2.22 | Keep `sqlalchemy==2.0.57` (pinned) |
| 9 | `/reload` Internal Server Error or "alias not found" | MLflow down, or no model registered, or wrong MLflow URL | Start MLflow, run `src.register`, check `MLFLOW_TRACKING_URI`, then `/reload`. The response now shows the real reason |
| 10 | Dashboard: `{"detail":"model not loaded"}` | API started before a model existed, or points at another MLflow | Register a model on the same MLflow the API uses, then POST `/reload` |
| 11 | Dashboard refused on 8501 | Streamlit not running | Run `streamlit run dashboard/app.py` |
| 12 | XGBoost fails in Docker | `libgomp1` missing in slim image | Already installed in `Dockerfile.api` |
| 13 | Retrain says KEPT although drift is high | RMSE noisy on a small holdout | Decision uses MAE (`DECISION_METRIC` in `src/config.py`) |

Quick checks: `lsof -i :5001` (who uses a port), `docker compose ps`, `docker compose logs mlflow`.
