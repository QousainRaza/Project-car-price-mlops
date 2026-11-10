"""Step 10: retrain when drift is high, and promote the new model ONLY if it is better.

Rule: score the new model and the current production model on the same holdout rows
(the last 20% of the newest batch, which neither model trained on).
Run:  python -m src.retrain          (new data = batch 3)
      python -m src.retrain 3 --force
"""
import json
import sys
from datetime import datetime, timezone

import mlflow
import pandas as pd
import requests
from xgboost import XGBRegressor

from src.config import (API_URL, BEST_PARAMS, DECISION_METRIC, EXPERIMENT, FEATURES, MLFLOW_URI,
                        MODEL_NAME, PROMOTE_MARGIN, RECENT_WEIGHT, REPORTS_DIR, TARGET)
from src.drift import check
from src.modeling import build, evaluate, load_batch
from src.register import promote


def _save(result):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    result["time"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (REPORTS_DIR / "retrain_log.json").write_text(json.dumps(result, indent=2))


def run(new_batch=3, force=False, reload_api=True):
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    drift = check(reference_batch=0, new_batch=new_batch)
    if not drift["drift_detected"] and not force:
        result = {"decision": "SKIPPED", "reason": f"no drift ({drift['share_drifted']:.0%} of columns)"}
        _save(result)
        return result

    old_batches = [load_batch(i) for i in range(new_batch)]
    new = load_batch(new_batch)
    cut = int(len(new) * 0.8)
    new_train = new.iloc[:cut]
    train = pd.concat(old_batches + [new_train], ignore_index=True)
    holdout = new.iloc[cut:]
    # old rows count 1x, rows from the newest batch count RECENT_WEIGHT x
    weights = [1.0] * (len(train) - len(new_train)) + [float(RECENT_WEIGHT)] * len(new_train)

    with mlflow.start_run(run_name=f"retrain_batch{new_batch}") as run_:
        candidate = build(XGBRegressor(**BEST_PARAMS)).fit(
            train[FEATURES], train[TARGET], model__sample_weight=weights)
        new_m = evaluate(candidate, holdout)

        try:                                    # current production model (may not exist yet)
            prod = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@production")
            old_m = evaluate(prod, holdout)
        except Exception as e:                  # noqa: BLE001
            print("no production model to compare with:", e)
            old_m = {"rmse": float("inf"), "mae": float("inf"), "r2": float("nan")}

        mlflow.log_params({"source": "retrain", "new_batch": new_batch,
                           "recent_weight": RECENT_WEIGHT, **BEST_PARAMS})
        mlflow.log_metrics({"holdout_rmse": new_m["rmse"], "holdout_mae": new_m["mae"],
                            "holdout_r2": new_m["r2"], "old_holdout_mae": old_m["mae"]})
        mlflow.sklearn.log_model(candidate, artifact_path="model", input_example=holdout[FEATURES].head(3))

        better = new_m[DECISION_METRIC] < old_m[DECISION_METRIC] * PROMOTE_MARGIN
        result = {"decision": "PROMOTED" if better else "KEPT", "metric": DECISION_METRIC,
                  "old_score": old_m[DECISION_METRIC], "new_score": new_m[DECISION_METRIC],
                  "old_rmse": old_m["rmse"], "new_rmse": new_m["rmse"],
                  "drift_share": drift["share_drifted"],
                  "holdout_rows": int(len(holdout)), "train_rows": int(len(train))}
        if better:
            result["version"] = promote(run_.info.run_id)
            if reload_api:
                try:
                    requests.post(f"{API_URL}/reload", timeout=30).raise_for_status()
                    result["api_reloaded"] = True
                except Exception as e:          # noqa: BLE001
                    result["api_reloaded"] = f"failed: {e}"
    _save(result)
    return result


if __name__ == "__main__":
    batch = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 3
    res = run(new_batch=batch, force="--force" in sys.argv)
    if res["decision"] == "PROMOTED":
        print(f"PROMOTED v{res['version']}: {res['metric'].upper()} {res['old_score']:,.0f} -> {res['new_score']:,.0f}")
    elif res["decision"] == "KEPT":
        print(f"KEPT old model: {res['metric'].upper()} old {res['old_score']:,.0f}, candidate {res['new_score']:,.0f}")
    else:
        print("SKIPPED:", res["reason"])
