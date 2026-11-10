"""One place for paths, column names and settings."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "cars_raw.csv"
BATCH_DIR = ROOT / "data" / "batches"
SAMPLE_CSV = ROOT / "data" / "sample" / "cars_sample.csv"
REPORTS_DIR = ROOT / "reports"

# raw CSV column  ->  standard name.  (In the raw file, "brand" is really the full listing title.)
RAW_TO_STD = {
    "brand": "title",
    "price": "price",
    "year": "year",
    "mileage": "mileage",
    "fuel": "fuel",
    "transmission": "transmission",
    "city": "city",
    "engine_cc": "engine_cc",
}

TARGET = "price"
NUMERIC = ["car_age", "mileage", "engine_cc"]
CATEGORICAL = ["brand", "car_model", "city", "fuel", "transmission"]
FEATURES = CATEGORICAL + NUMERIC

CURRENT_YEAR = 2026
YEAR_MIN, YEAR_MAX = 1970, 2026
ENGINE_MIN, ENGINE_MAX = 300, 8000

MODEL_NAME = "car-price"
EXPERIMENT = "car-price-regression"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# Settings used when retraining
BEST_PARAMS = dict(n_estimators=400, max_depth=10, learning_rate=0.08,
                   subsample=0.9, colsample_bytree=0.9, n_jobs=4, random_state=42)
DRIFT_THRESHOLD = 0.30      # alarm if more than 30% of columns drifted
DECISION_METRIC = "mae"     # best-model and promotion decisions use MAE: it is not thrown off by a few very expensive cars
PROMOTE_MARGIN = 0.99       # new model must have MAE below 99% of the old one
RECENT_WEIGHT = 10          # when retraining, rows from the newest batch count 10x (the market moved)
