# Used-Car Price MLOps (Pakistan)

Predict a used car's price from a CSV, and run it like a product: track, register, serve, monitor, retrain.
Note: batches are SIMULATED from one CSV (batch 3 = newer cars, prices x1.25 to demo drift).

## Setup
    python -m venv venv && source venv/bin/activate
    pip install -r requirements.txt

## Run order (local)
    mlflow server --host 127.0.0.1 --port 5000 --serve-artifacts &   # tracking server
    python -m src.make_batches        # clean + split + reports
    python -m src.train               # 4 runs -> MLflow
    python -m src.register            # best run -> alias "production"
    uvicorn api.main:app --port 8000 &
    streamlit run dashboard/app.py    # http://localhost:8501
    pytest -q
    python -m src.quality_gate

## Monitor + retrain
    python -m src.drift 3             # Evidently report -> reports/drift.html
    python -m src.retrain             # drift -> retrain -> promote only if MAE better -> API reload
    python flows/weekly_pipeline.py   # same, as a Prefect flow

## Docker
Start Docker Desktop, then: `docker compose up --build`
MLflow :5000, API :8000, Dashboard :8501. Containers reach each other by service name.
Train against the container server: `MLFLOW_TRACKING_URI=http://127.0.0.1:5000 python -m src.train && python -m src.register && curl -X POST localhost:8000/reload`

## Demo script (5 min)
1. MLflow UI: compare runs. 2. Dashboard: estimate a price. 3. `python -m src.drift 3`: show drift report. 4. `python -m src.retrain`: PROMOTED, version 2 live. 5. Show CI.

## Troubleshooting
- `/reload` error: the response now contains the real reason; check MLflow is up.
- MLflow crash with FallbackAsyncAdaptedQueuePool: keep `sqlalchemy<2.1`.
- Port 5000 busy: stop the local MLflow or change the compose port.
- Pandera errors: re-run `python -m src.make_batches`.
