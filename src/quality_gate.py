"""Step 9: CI quality gate. Fails (exit code 1) if the model is not clearly better than guessing the mean.
Uses the small committed sample, so it needs no MLflow server and no big data.
Run:  python -m src.quality_gate
"""
import sys

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from src.config import FEATURES, SAMPLE_CSV, TARGET
from src.features import add_features
from src.modeling import build
from src.validate import validate

MAX_RATIO = 0.70      # model RMSE must be below 70% of the dummy RMSE


def main():
    df = add_features(validate(pd.read_csv(SAMPLE_CSV)))
    Xtr, Xte, ytr, yte = train_test_split(df[FEATURES], df[TARGET], test_size=0.3, random_state=0)

    model = build(XGBRegressor(n_estimators=150, max_depth=8, learning_rate=0.1, n_jobs=2, random_state=0)).fit(Xtr, ytr)
    dummy = DummyRegressor(strategy="mean").fit(Xtr, ytr)

    rmse = lambda m: mean_squared_error(yte, m.predict(Xte)) ** 0.5
    ratio = rmse(model) / rmse(dummy)
    print(f"model RMSE is {ratio:.0%} of the dummy RMSE (limit {MAX_RATIO:.0%})")
    sys.exit(0 if ratio < MAX_RATIO else 1)


if __name__ == "__main__":
    main()
