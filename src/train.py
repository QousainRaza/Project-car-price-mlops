"""Step 4: train models and log every run to MLflow.

Train on batch_0, test on batch_1 (data the model has never seen).
Run:  python -m src.train
"""
import mlflow
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from src.config import EXPERIMENT, FEATURES, MLFLOW_URI, TARGET
from src.modeling import build, evaluate, load_batch


def candidates():
    xgb = dict(n_jobs=4, random_state=42)
    return {
        "ridge": Ridge(alpha=1.0),                                                  # simple baseline
        "xgb_depth8": XGBRegressor(n_estimators=300, max_depth=8, learning_rate=0.1, **xgb),
        "xgb_depth12": XGBRegressor(n_estimators=300, max_depth=12, learning_rate=0.1, **xgb),
        "xgb_slow": XGBRegressor(n_estimators=600, max_depth=8, learning_rate=0.05, **xgb),
    }


def main():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)
    train, test = load_batch(0), load_batch(1)
    print(f"train rows: {len(train)}   test rows: {len(test)}")

    for name, model in candidates().items():
        with mlflow.start_run(run_name=name):
            pipe = build(model).fit(train[FEATURES], train[TARGET])
            metrics = evaluate(pipe, test)
            mlflow.log_params({"model": name, "log_target": True, "train_batch": 0, "test_batch": 1,
                               **model.get_params()})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipe, artifact_path="model", input_example=test[FEATURES].head(3))
            print(f"{name:12s} rmse={metrics['rmse']:>12,.0f}  mae={metrics['mae']:>10,.0f}  r2={metrics['r2']:.3f}")


if __name__ == "__main__":
    main()
