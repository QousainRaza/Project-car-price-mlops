"""Turn the raw CSV into clean, standard columns. Used by make_batches.py and by tests."""
import pandas as pd

from src.config import (ENGINE_MAX, ENGINE_MIN, RAW_TO_STD, YEAR_MAX, YEAR_MIN)

TWO_WORD_MAKES = {"Land Rover", "Range Rover", "Mercedes Benz"}


def split_title(title):
    """'Suzuki Alto VXL AGS' -> ('Suzuki', 'Suzuki Alto')."""
    words = str(title).split()
    if not words:
        return "Unknown", "Unknown"
    two = " ".join(words[:2])
    brand = two if two in TWO_WORD_MAKES else words[0]
    rest = words[len(brand.split()):]
    if not rest:
        return brand, brand
    # keep short model names like "N Box" together
    name = " ".join(rest[:2]) if len(rest[0]) <= 2 and len(rest) > 1 else rest[0]
    return brand, f"{brand} {name}"


def clean_engine(series):
    """'1000 cc' -> 1000.0 (bad values become NaN)."""
    return pd.to_numeric(
        series.astype(str).str.replace("cc", "", case=False).str.replace(",", "").str.strip(),
        errors="coerce",
    )


def standardize(raw):
    """Return (clean_df, log). `log` says how many rows each rule removed."""
    df = raw.rename(columns={v: k for k, v in RAW_TO_STD.items()})
    df = df[list(RAW_TO_STD.keys())].copy()          # drops extra columns such as 'Unnamed: 0'
    df["title"] = df["brand"]                          # raw 'brand' is the full listing title
    parts = df["title"].map(split_title)
    df["brand"] = parts.str[0]
    df["car_model"] = parts.str[1]
    df["engine_cc"] = clean_engine(df["engine_cc"])
    for col in ("price", "year", "mileage"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    log = {"rows_in": int(len(df))}

    def step(name, mask):
        nonlocal df
        log[name] = int((~mask).sum())
        df = df[mask]

    step("missing_values", df[["price", "year", "mileage", "engine_cc"]].notna().all(axis=1))
    step("year_out_of_range", df["year"].between(YEAR_MIN, YEAR_MAX))
    step("engine_out_of_range", df["engine_cc"].between(ENGINE_MIN, ENGINE_MAX))
    step("bad_price_or_mileage", (df["price"] > 0) & df["mileage"].between(0, 1_000_000))
    before = len(df)
    df = df.drop_duplicates(subset=["title", "price", "year", "mileage", "city", "engine_cc", "fuel", "transmission"])
    log["duplicates"] = int(before - len(df))

    df = df.drop(columns=["title"])
    df["year"] = df["year"].astype(int)
    df["price"] = df["price"].astype(float)
    df = df[["price", "year", "mileage", "brand", "car_model", "city", "engine_cc", "fuel", "transmission"]]
    log["rows_out"] = int(len(df))
    log["rows_removed"] = log["rows_in"] - log["rows_out"]
    return df.reset_index(drop=True), log
