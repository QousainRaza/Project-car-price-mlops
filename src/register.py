"""Step 5: register the best run and mark it as 'production'.
Run:  python -m src.register
"""
import mlflow
from mlflow.tracking import MlflowClient

from src.config import DECISION_METRIC, EXPERIMENT, MLFLOW_URI, MODEL_NAME


def promote(run_id):
    """Register a run's model as a new version and move the 'production' alias to it."""
    client = MlflowClient()
    mv = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
    client.set_registered_model_alias(MODEL_NAME, "production", mv.version)
    live = client.get_model_version_by_alias(MODEL_NAME, "production")   # check it really worked
    return live.version


def register_best():
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        raise SystemExit(f"No experiment '{EXPERIMENT}' on {MLFLOW_URI}. Run: python -m src.train")
    runs = client.search_runs([exp.experiment_id], filter_string=f"metrics.{DECISION_METRIC} > 0",
                              order_by=[f"metrics.{DECISION_METRIC} ASC"], max_results=1)
    if not runs:
        raise SystemExit(f"No runs with a {DECISION_METRIC} metric yet. Run: python -m src.train")
    best = runs[0]
    version = promote(best.info.run_id)
    print(f"production -> {MODEL_NAME} v{version}  ({best.data.tags.get('mlflow.runName')}, "
          f"{DECISION_METRIC} {best.data.metrics[DECISION_METRIC]:,.0f})  on {MLFLOW_URI}")


if __name__ == "__main__":
    register_best()
