# Used-Car Price MLOps (Pakistan)

Predict a used car's price from a CSV, and run it like a product:
**clean data → train + track → register → serve (API + dashboard) → monitor drift → retrain → CI → Docker.**

> Batches are SIMULATED from one CSV (batch 3 = newer cars, prices x1.25) to demo drift. Say this in class.

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
