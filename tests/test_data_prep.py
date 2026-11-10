import pandas as pd

from src.data_prep import clean_engine, split_title, standardize


def test_split_title():
    assert split_title("Suzuki Alto VXL AGS") == ("Suzuki", "Suzuki Alto")
    assert split_title("Honda N Box Custom GL") == ("Honda", "Honda N Box")
    assert split_title("Land Rover Defender") == ("Land Rover", "Land Rover Defender")


def test_clean_engine():
    out = clean_engine(pd.Series(["1000 cc", "1,300 cc", "bad"]))
    assert out.iloc[0] == 1000 and out.iloc[1] == 1300 and pd.isna(out.iloc[2])


def test_standardize_removes_bad_rows():
    row = dict(brand="Toyota Corolla GLi", price=2_500_000, year=2015, mileage=80000.0,
               fuel="Petrol", transmission="Manual", city="Lahore", engine_cc="1300 cc")
    raw = pd.DataFrame([
        {"Unnamed: 0": 0, **row},
        {"Unnamed: 0": 1, **row},                                   # duplicate of row 0
        {"Unnamed: 0": 2, **{**row, "year": 1900}},                 # junk year
        {"Unnamed: 0": 3, **{**row, "engine_cc": "10000 cc"}},      # engine too big
        {"Unnamed: 0": 4, **{**row, "engine_cc": None, "year": 2020}},   # missing engine
    ])
    df, log = standardize(raw)
    assert len(df) == 1
    assert df.loc[0, "brand"] == "Toyota" and df.loc[0, "car_model"] == "Toyota Corolla"
    assert log["rows_in"] == 5 and log["rows_out"] == 1
