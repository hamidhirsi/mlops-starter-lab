"""Check your work at the end of an exercise.

Run it with the exercise number, for example after the tracking exercise:

    uv run python scripts/check.py 4

The number matches the exercise page. Sections 4 to 11 have checks; the first
three exercises are setup and reading, and `scripts/check_setup.py` covers
those. The script only reads and reports: it never fixes or changes anything,
so a pass means your own work is complete.
"""
import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

failures = 0


def ok(message):
    print(f"  PASS  {message}")


def bad(message, hint):
    global failures
    failures += 1
    print(f"  FAIL  {message}")
    print(f"        Hint: {hint}")


def need_tracking_uri():
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        bad(
            "MLFLOW_TRACKING_URI is not set",
            "run `export MLFLOW_TRACKING_URI=http://localhost:5001` first",
        )
    return uri


def http_get(url, timeout=10):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError):
        return None, ""


def run_command(args, timeout=300):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def check_4():
    """Tracking: four runs in the housing experiment, each fully recorded."""
    if not need_tracking_uri():
        return
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    experiment = client.get_experiment_by_name("housing")
    if experiment is None:
        bad(
            "there is no experiment called `housing`",
            "your training script should call mlflow.set_experiment('housing') "
            "before starting a run",
        )
        return
    ok("the `housing` experiment exists")

    runs = client.search_runs(experiment_ids=[experiment.experiment_id], max_results=100)
    if len(runs) < 4:
        bad(
            f"the experiment has {len(runs)} runs, the exercise trains four",
            "run `uv run python -m src.train --n-estimators N` "
            "for N of 25, 50, 100, and 200",
        )
        return
    ok(f"it has {len(runs)} runs")

    complete = 0
    for run in runs:
        has_param = "n_estimators" in run.data.params
        has_metric = "rmse" in run.data.metrics
        models = client.search_logged_models(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"source_run_id='{run.info.run_id}'",
        )
        if has_param and has_metric and models:
            complete += 1
    if complete >= 4:
        ok(f"{complete} runs carry the n_estimators parameter, the rmse metric, and a model")
    else:
        bad(
            f"only {complete} runs carry all three of: n_estimators, rmse, and a logged model",
            "check your tracking block logs the parameter, the metric, "
            "and calls mlflow.sklearn.log_model",
        )


def check_5():
    """Registry: the model is registered, aliased, and loadable by name."""
    if not need_tracking_uri():
        return
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    try:
        client.get_registered_model("housing-predictor")
        ok("`housing-predictor` is registered")
    except MlflowException:
        bad(
            "no registered model called `housing-predictor`",
            "run `uv run python -m src.register`",
        )
        return

    try:
        version = client.get_model_version_by_alias("housing-predictor", "production")
        ok(f"the `production` alias points at version {version.version}")
    except MlflowException:
        bad(
            "no `production` alias on the model",
            "your register script should call set_registered_model_alias",
        )
        return

    if not Path("src/predict.py").exists():
        bad("src/predict.py does not exist", "create it as shown in the exercise")
        return
    result = run_command([sys.executable, "-m", "src.predict"])
    if result and result.returncode == 0 and "Prediction" in result.stdout:
        ok("src/predict.py runs and prints a prediction")
    else:
        detail = (result.stderr.strip().splitlines() or ["no output"])[-1] if result else "it timed out"
        bad(f"src/predict.py failed ({detail})", "run it yourself to see the full error")


def check_6():
    """Feature store: definitions applied, features materialised into Redis."""
    for path in ("feature_repo/feature_store.yaml", "feature_repo/feature_repo.py"):
        if Path(path).exists():
            ok(f"{path} exists")
        else:
            bad(f"{path} does not exist", "create it as shown in the exercise")
            return

    from feast import FeatureStore

    try:
        store = FeatureStore(repo_path="feature_repo")
        store.get_feature_view("district_features")
        ok("the Feast registry knows the `district_features` view")
    except Exception as exc:
        bad(
            f"Feast could not find the `district_features` view ({type(exc).__name__})",
            "run `uv run feast apply` from inside feature_repo/",
        )
        return

    from redis import Redis

    try:
        keys = Redis(host="localhost", port=6379).dbsize()
    except Exception:
        bad(
            "could not reach Redis on localhost:6379",
            "check the redis service with `docker compose ps`",
        )
        return
    if keys > 0:
        ok(f"Redis holds {keys} keys of feature data")
    else:
        bad(
            "Redis is empty, nothing has been materialised",
            "run `uv run feast materialize-incremental "
            "$(date -u +'%Y-%m-%dT%H:%M:%S')` from inside feature_repo/",
        )


def check_7():
    """Training infrastructure: the checkpoint option exists and produced a file."""
    result = run_command([sys.executable, "-m", "src.train", "--help"], timeout=60)
    if result and "--checkpoint-every" in result.stdout:
        ok("src/train.py accepts --checkpoint-every")
    else:
        bad(
            "src/train.py does not accept --checkpoint-every",
            "add the option and the warm_start batch loop from the exercise",
        )
        return
    if Path("checkpoint.pkl").exists():
        ok("checkpoint.pkl exists")
    else:
        bad(
            "checkpoint.pkl does not exist",
            "run `uv run python -m src.train --n-estimators 200 --checkpoint-every 50`",
        )


def check_8():
    """Serving: the API answers ready and exposes the prediction metrics."""
    status, _ = http_get("http://localhost:8000/ready")
    if status == 200:
        ok("the service answers /ready with 200 on port 8000")
    else:
        bad(
            "nothing answered 200 at http://localhost:8000/ready",
            "start it with `uv run uvicorn src.serve:app --port 8000` "
            "and wait for the model to load",
        )
        return
    status, body = http_get("http://localhost:8000/metrics")
    if status == 200 and "prediction_requests_total" in body:
        ok("/metrics exposes prediction_requests_total")
    else:
        bad(
            "/metrics does not expose prediction_requests_total",
            "check the Counter name and that the metrics app is mounted at /metrics",
        )


def check_9():
    """CI: the three test files pass and the workflow file parses."""
    missing = [
        p
        for p in ("tests/test_data.py", "tests/test_model.py", "tests/test_pipeline.py")
        if not Path(p).exists()
    ]
    if missing:
        bad(f"missing test files: {', '.join(missing)}", "create them as shown in the exercise")
        return
    ok("all three test files exist")

    result = run_command(["uv", "run", "pytest", "-q"], timeout=600)
    if result and result.returncode == 0:
        summary = (result.stdout.strip().splitlines() or ["passed"])[-1]
        ok(f"pytest passes ({summary})")
    else:
        bad("pytest fails", "run `uv run pytest -v` to see which test and why")

    workflow = Path(".github/workflows/ml-ci.yml")
    if not workflow.exists():
        bad(".github/workflows/ml-ci.yml does not exist", "create it as shown in the exercise")
        return
    import yaml

    try:
        parsed = yaml.safe_load(workflow.read_text())
        # YAML reads the bare `on:` trigger key as the boolean True.
        if isinstance(parsed, dict) and "jobs" in parsed and (True in parsed or "on" in parsed):
            ok("the workflow file parses and has triggers and jobs")
        else:
            bad(
                "the workflow file parses but is missing `on:` or `jobs:`",
                "compare it against the exercise page",
            )
    except yaml.YAMLError as exc:
        bad(f"the workflow file is not valid YAML ({exc})", "check the indentation")


def check_10():
    """Monitoring: the drift summary exists and Prometheus sees your service."""
    if not Path("src/drift_check.py").exists():
        bad("src/drift_check.py does not exist", "create it as shown in the exercise")
        return
    ok("src/drift_check.py exists")

    summary_path = Path("drift_summary.json")
    if not summary_path.exists():
        bad(
            "drift_summary.json does not exist",
            "run `uv run python -m src.drift_check`",
        )
        return
    try:
        summary = json.loads(summary_path.read_text())
    except json.JSONDecodeError:
        bad("drift_summary.json is not valid JSON", "re-run `uv run python -m src.drift_check`")
        return
    expected = {"share_of_drifted_columns", "dataset_drift"}
    if expected.issubset(summary):
        ok(
            "drift_summary.json has both keys "
            f"(share_of_drifted_columns is {summary['share_of_drifted_columns']})"
        )
    else:
        bad(
            f"drift_summary.json is missing {', '.join(sorted(expected - set(summary)))}",
            "compare your summary block against the exercise page",
        )

    status, body = http_get("http://localhost:9090/api/v1/targets")
    if status != 200:
        bad(
            "could not reach Prometheus on localhost:9090",
            "check the prometheus service with `docker compose ps`",
        )
        return
    up = any(
        t.get("labels", {}).get("job") == "serving" and t.get("health") == "up"
        for t in json.loads(body).get("data", {}).get("activeTargets", [])
    )
    if up:
        ok("Prometheus reports the `serving` target as up")
    else:
        bad(
            "Prometheus does not see your serving API as up",
            "keep `uv run uvicorn src.serve:app --port 8000` running, "
            "then check Status and then Targets at http://localhost:9090",
        )


def check_11():
    """Pipeline: the DAG imports cleanly and has run to success."""
    if not Path("airflow/dags/retrain_on_drift.py").exists():
        bad("airflow/dags/retrain_on_drift.py does not exist", "create it as shown in the exercise")
        return
    ok("the DAG file exists")

    errors = run_command(
        ["docker", "compose", "exec", "-T", "airflow-scheduler",
         "airflow", "dags", "list-import-errors"],
        timeout=120,
    )
    if errors is None:
        bad(
            "could not run a command in the scheduler container",
            "check the airflow-scheduler service with `docker compose ps`",
        )
        return
    if "retrain_on_drift" in errors.stdout:
        bad(
            "the DAG has an import error inside the scheduler",
            "the same output shows the traceback: read it, fix the DAG file",
        )
        return
    listed = run_command(
        ["docker", "compose", "exec", "-T", "airflow-scheduler",
         "airflow", "dags", "list"],
        timeout=120,
    )
    if listed and "retrain_on_drift" in listed.stdout:
        ok("the scheduler imports the DAG cleanly")
    else:
        bad(
            "the scheduler does not list retrain_on_drift",
            "give it a minute to scan the dags folder, then check "
            "`docker compose exec airflow-scheduler airflow dags list`",
        )
        return

    runs = run_command(
        ["docker", "compose", "exec", "-T", "airflow-scheduler",
         "airflow", "dags", "list-runs", "-d", "retrain_on_drift"],
        timeout=120,
    )
    if runs and "success" in runs.stdout:
        ok("the DAG has at least one successful run")
    else:
        bad(
            "the DAG has no successful run yet",
            "unpause and trigger it as shown in the exercise, then watch it "
            "at http://localhost:8081",
        )


CHECKS = {
    4: check_4, 5: check_5, 6: check_6, 7: check_7,
    8: check_8, 9: check_9, 10: check_10, 11: check_11,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("section", type=int, choices=sorted(CHECKS),
                        help="the exercise number to check")
    args = parser.parse_args()

    print(f"Checking exercise {args.section}\n")
    CHECKS[args.section]()
    print()
    if failures == 0:
        print("All checks for this exercise passed.")
        return 0
    print("Not there yet. The hints above say what is missing.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
