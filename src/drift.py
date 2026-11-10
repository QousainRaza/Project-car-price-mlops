"""Step 10: data drift check with Evidently (version 0.4.x).
Compares a new batch with the training batch, column by column.
Run:  python -m src.drift            (checks batch 3)
      python -m src.drift 1          (checks batch 1)
"""
import json
import sys

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from src.config import BATCH_DIR, DRIFT_THRESHOLD, FEATURES, REPORTS_DIR
from src.features import add_features


def check(reference_batch=0, new_batch=3):
    ref = add_features(pd.read_csv(BATCH_DIR / f"batch_{reference_batch}.csv"))[FEATURES]
    cur = add_features(pd.read_csv(BATCH_DIR / f"batch_{new_batch}.csv"))[FEATURES]

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report.save_html(str(REPORTS_DIR / "drift.html"))
    result = report.as_dict()["metrics"][0]["result"]
    share = float(result["share_of_drifted_columns"])
    out = {
        "new_batch": new_batch,
        "share_drifted": share,
        "drifted_columns": int(result["number_of_drifted_columns"]),
        "total_columns": int(result["number_of_columns"]),
        "drift_detected": share > DRIFT_THRESHOLD,
    }
    (REPORTS_DIR / "drift_summary.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    batch = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    print(check(new_batch=batch))
