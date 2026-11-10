import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.config import FEATURES, TARGET
from src.features import add_features
from src.modeling import build, evaluate


def tiny_data(n=400):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "year": rng.integers(2005, 2025, n),
        "mileage": rng.uniform(5_000, 200_000, n),
        "engine_cc": rng.choice([660, 1000, 1300, 1600], n).astype(float),
        "brand": rng.choice(["Toyota", "Suzuki"], n),
        "car_model": rng.choice(["Toyota Corolla", "Suzuki Alto"], n),
        "city": rng.choice(["Lahore", "Karachi"], n),
        "fuel": rng.choice(["Petrol", "Diesel"], n),
        "transmission": rng.choice(["Manual", "Automatic"], n),
    })
    df = add_features(df)
    df["price"] = 3_000_000 - df["car_age"] * 80_000 - df["mileage"] * 3 + df["engine_cc"] * 500
    return df


def test_pipeline_trains_and_predicts():
    df = tiny_data()
    pipe = build(Ridge()).fit(df[FEATURES], df[TARGET])
    metrics = evaluate(pipe, df)
    assert set(metrics) == {"rmse", "mae", "r2"}
    assert metrics["r2"] > 0.5


def test_unknown_category_does_not_crash():
    df = tiny_data()
    pipe = build(Ridge()).fit(df[FEATURES], df[TARGET])
    new = df.head(2).copy()
    new["brand"] = "NeverSeenBefore"
    assert len(pipe.predict(new[FEATURES])) == 2
