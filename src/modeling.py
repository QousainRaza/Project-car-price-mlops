"""Shared model code (no MLflow in here, so tests and CI can import it without a server)."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import BATCH_DIR, CATEGORICAL, FEATURES, NUMERIC, TARGET
from src.features import add_features
from src.validate import validate


def build(model, log_target=True):
    """Pipeline: one-hot the categories, scale the numbers, then the model.
    Prices are very skewed, so by default we learn log(price) and convert back."""
    prep = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20), CATEGORICAL),
        ("num", StandardScaler(), NUMERIC),
    ])
    reg = TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1) if log_target else model
    return Pipeline([("prep", prep), ("model", reg)])


def evaluate(pipe, df):
    """Metrics on the ORIGINAL price scale."""
    pred = pipe.predict(df[FEATURES])
    return {
        "rmse": float(mean_squared_error(df[TARGET], pred) ** 0.5),
        "mae": float(mean_absolute_error(df[TARGET], pred)),
        "r2": float(r2_score(df[TARGET], pred)),
    }


def load_batch(i):
    """Read a batch, validate it (stops on bad data), add features."""
    return add_features(validate(pd.read_csv(BATCH_DIR / f"batch_{i}.csv")))
