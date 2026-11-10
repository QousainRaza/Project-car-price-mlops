"""One feature function, used in training AND in the API (avoids training-serving skew)."""
from src.config import CURRENT_YEAR


def add_features(df):
    df = df.copy()
    df["car_age"] = (CURRENT_YEAR - df["year"]).clip(lower=0)
    return df
