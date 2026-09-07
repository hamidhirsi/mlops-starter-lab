# Reference solution for exercise 11, production.
import json
from datetime import datetime

from airflow import DAG
from airflow.exceptions import AirflowSkipException
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

DRIFT_THRESHOLD = 0.2


def check_drift():
    summary = json.load(open("/opt/project/drift_summary.json"))
    share = summary["share_of_drifted_columns"]
    print(f"drift share: {share}")
    if share < DRIFT_THRESHOLD:
        raise AirflowSkipException("no significant drift, nothing to do")


with DAG(
    dag_id="retrain_on_drift",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    # One run at a time. Unpausing a daily DAG immediately starts a catch up
    # run for the last missed interval, and the exercise then triggers a manual
    # run seconds later. Without this line the two run at once, the two
    # trainings compete for memory, and one gets killed. With it they queue.
    max_active_runs=1,
) as dag:
    check = PythonOperator(task_id="check_drift", python_callable=check_drift)
    drift = BashOperator(task_id="run_drift_check",
                         bash_command="cd /opt/project && python -m src.drift_check")
    retrain = BashOperator(task_id="retrain_model",
                           bash_command="cd /opt/project && python -m src.train --n-estimators 250")
    promote = BashOperator(task_id="promote_model",
                           bash_command="cd /opt/project && python -m src.register")

    drift >> check >> retrain >> promote
