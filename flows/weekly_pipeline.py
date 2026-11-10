"""Step 10b: Prefect runs the drift check and the retrain job on a schedule.

Run once now:        python flows/weekly_pipeline.py now
Run on a schedule:   python flows/weekly_pipeline.py        (every Monday 03:00)
Web page:            prefect server start      then open http://127.0.0.1:4200
                     prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api   (once)
Run from the project root. This only runs while your computer (or server) is on.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prefect import flow, task  # noqa: E402

from src.drift import check  # noqa: E402
from src.retrain import run as retrain_run  # noqa: E402


@task(retries=2, retry_delay_seconds=30)
def drift_task(batch):
    return check(new_batch=batch)


@task
def retrain_task(batch):
    return retrain_run(new_batch=batch)


@flow(log_prints=True)
def weekly_pipeline(new_batch: int = 3):
    result = drift_task(new_batch)
    print("drift result:", result)
    if result["drift_detected"]:
        print("decision:", retrain_task(new_batch))
    else:
        print("no drift, nothing to do")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "now":
        weekly_pipeline(3)
    else:
        weekly_pipeline.serve(name="weekly", cron="0 3 * * 1")     # every Monday, 3am
