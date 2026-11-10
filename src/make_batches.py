"""Step 2: clean the raw CSV and cut it into 4 batches that 'arrive' one after another.

batch_0  60%  training data
batch_1  15%  new data, looks the same
batch_2  15%  new data, looks the same
batch_3  10%  the market changed: newer cars only, prices x1.25  (this is the drift demo)

The batches are SIMULATED from one file. Say that in your class.
Run:  python -m src.make_batches
"""
import json

import pandas as pd

from src.config import BATCH_DIR, RAW_CSV, REPORTS_DIR, SAMPLE_CSV
from src.data_prep import standardize


def write_reports(df, log):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "cleaning_log.json").write_text(json.dumps(log, indent=2))

    # options for the web app's drop-downs
    top_brands = df["brand"].value_counts().head(40).index.tolist()
    models_by_brand = {}
    for brand in top_brands:
        counts = df.loc[df["brand"] == brand, "car_model"].value_counts()
        models_by_brand[brand] = counts[counts >= 5].head(40).index.tolist() or [f"{brand}"]
    choices = {
        "brands": top_brands,
        "models_by_brand": models_by_brand,
        "cities": df["city"].value_counts().head(30).index.tolist(),
        "fuels": sorted(df["fuel"].unique().tolist()),
        "transmissions": sorted(df["transmission"].unique().tolist()),
    }
    (REPORTS_DIR / "choices.json").write_text(json.dumps(choices, indent=2))

    # small numbers for the "Market" tab
    by_year = df.groupby("year")["price"].agg(["median", "count"])
    by_year = by_year[(by_year["count"] >= 30) & (by_year.index >= 2000)]
    by_brand = df.groupby("brand")["price"].agg(["median", "count"])
    by_brand = by_brand[by_brand["count"] >= 200].sort_values("median", ascending=False).head(12)
    eda = {
        "median_price_by_year": {str(k): float(v) for k, v in by_year["median"].items()},
        "median_price_by_brand": {k: float(v) for k, v in by_brand["median"].items()},
    }
    (REPORTS_DIR / "eda_summary.json").write_text(json.dumps(eda, indent=2))


def main():
    raw = pd.read_csv(RAW_CSV)
    df, log = standardize(raw)
    print(f"cleaning removed {log['rows_removed']} of {log['rows_in']} rows")
    print({k: v for k, v in log.items() if k not in ("rows_in", "rows_out", "rows_removed")})

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    SAMPLE_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.sample(3000, random_state=1).to_csv(SAMPLE_CSV, index=False)     # small file for CI

    n = len(df)
    cuts = [0, int(0.60 * n), int(0.75 * n), int(0.90 * n), n]
    b0, b1, b2, b3 = (df.iloc[cuts[i]:cuts[i + 1]] for i in range(4))
    b3 = b3[b3["year"] >= b3["year"].quantile(0.75)].copy()             # newer cars only
    b3["price"] = b3["price"] * 1.25                                    # the market moved

    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    for i, b in enumerate([b0, b1, b2, b3]):
        b.to_csv(BATCH_DIR / f"batch_{i}.csv", index=False)
        print(f"batch_{i}: {len(b):>6} rows, mean price {b['price'].mean():,.0f}")

    write_reports(df, log)


if __name__ == "__main__":
    main()
