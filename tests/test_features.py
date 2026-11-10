import pandas as pd

from src.features import add_features


def test_car_age():
    df = pd.DataFrame({"year": [2016, 2026]})
    out = add_features(df)
    assert out["car_age"].tolist() == [10, 0]
    assert "car_age" not in df.columns        # the input is not changed
