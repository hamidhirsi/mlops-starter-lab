# mlops-starter-lab

The starter repository for the MLOps course. It gives you a working training
script, seed data, and a set of running services. Everything else, the
tracking, the registry, the feature store, the serving API, the tests, the
CI pipeline, the dashboards, and the retraining DAG, you build yourself,
one exercise at a time.

## What is in this repository

- `pyproject.toml`: project dependencies, pinned, managed with uv
- `docker-compose.yml`: the services you will build against (MLflow, Postgres, Redis, Prometheus, Grafana, Airflow)
- `docker/`: the Dockerfiles the compose file builds, with pinned versions
- `src/features.py`: the feature name contract shared by everything you will write
- `src/train.py`: a plain training script with no ML tooling attached, your starting point
- `scripts/seed_data.py`: generates the training data and the feature store source data
- `scripts/make_drift.py`: generates a deliberately drifted copy of the data
- `monitoring/prometheus.yml`: scrape config pointed at the serving API you will build
- `monitoring/grafana/`: Grafana provisioning with the Prometheus datasource ready and an empty dashboards folder
- `data/`: generated datasets land here (empty until you run the scripts)
- `feature_repo/data/`: feature store source data lands here; your Feast definitions will live in `feature_repo/`
- `airflow/dags/`: empty, your retraining DAG goes here
- `tests/`: empty, your tests go here
- `.github/workflows/`: empty, your CI pipeline goes here

## Prerequisites

- Python 3.11
- uv
- Docker with at least 6 GB of memory allocated

## Setup

```bash
git clone https://github.com/hamidhirsi/mlops-starter-lab.git
cd mlops-starter-lab

uv sync
uv run python scripts/seed_data.py
uv run python scripts/make_drift.py

# Linux only: run the Airflow containers as your own user, so files written
# by pipeline tasks stay editable by you. Mac and Windows skip this line.
echo "AIRFLOW_UID=$(id -u)" > .env

# --wait holds until every service reports healthy, so when this returns
# the lab is actually ready.
docker compose up -d --wait

export MLFLOW_TRACKING_URI=http://localhost:5001
```

The first `docker compose up` builds two images and can take around ten
minutes. After that it is cached.

Check that training works:

```bash
uv run python -m src.train
```

You should see an RMSE and a `model.pkl` file.

Then run the setup check. It looks at everything above and tells you what to
fix if something is off:

```bash
uv run python scripts/check_setup.py
```

## Services

| Service | Address | Login |
| --- | --- | --- |
| MLflow | http://localhost:5001 | none |
| Airflow | http://localhost:8081 | admin / admin |
| Grafana | http://localhost:3001 | admin / admin |
| Prometheus | http://localhost:9090 | none |
| Redis | localhost:6379 | none |
| Postgres (MLflow backend) | localhost:5432 | mlflow / mlflow |

Two ports are deliberately not the defaults: MLflow is on 5001 because macOS
reserves 5000 for AirPlay, and Airflow is on 8081 because many other tools
grab 8080.

## Where your work goes

The empty folders are yours. Each exercise adds one layer: tracking code into
`src/train.py`, registry and prediction scripts into `src/`, Feast definitions
into `feature_repo/`, a serving API into `src/`, tests into `tests/`, a
workflow into `.github/workflows/`, a dashboard JSON into
`monitoring/grafana/provisioning/dashboards/`, and finally a DAG into
`airflow/dags/`. Airflow mounts this whole repository at `/opt/project` and
has the project dependencies installed, so a DAG task can run the same
`python -m src.train` you run on the host.

## Reference solutions

The `solutions/` folder holds one reference solution per exercise,
snapshotted at the end of that exercise. Attempt the exercise first, pass its
check script, then compare your file against the reference with `diff`.
`solutions/README.md` explains the flow. Reading a solution before attempting
the exercise teaches very little, and nobody is grading the diff.

## Reference solutions

The `solutions/` folder holds one reference solution per exercise,
snapshotted at the end of that exercise. Attempt the exercise first, pass its
check script, then compare your file against the reference with `diff`. The
folder's own README explains the flow. Reading a solution before attempting
the exercise teaches very little; comparing after an honest attempt is where
it sticks.

## Troubleshooting

Before anything else, run the two check scripts. They diagnose most problems
and print the fix:

```bash
uv run python scripts/check_setup.py     # checks the whole lab
uv run python scripts/check.py 4         # checks one exercise, here number 4
```

Known problems, with the exact message you would see:

- `Bind for 0.0.0.0:5001 failed: port is already allocated` (any port):
  another program or container owns that port. Find it with `lsof -i :5001`
  or `docker ps`, stop it, then run `docker compose up -d --wait` again.

- A service shows `Restarting` in `docker compose ps`: it is crashing on
  startup. Read why with `docker compose logs <service-name> --tail 50`.

- The Airflow page at http://localhost:8081 does not load: the webserver
  takes a minute or two on a cold start. Watch it with
  `docker compose logs airflow-webserver --tail 20` and wait for
  `Listening at: http://0.0.0.0:8080`.

- `403 Forbidden` on every MLflow call from inside Airflow: the
  `MLFLOW_SERVER_ALLOWED_HOSTS` variable on the mlflow service was changed
  or removed. MLflow 3 rejects any Host header it has not been told to
  trust, and containers reach the server as `mlflow:5000`. Restore the line
  in `docker-compose.yml` and run `docker compose up -d mlflow`.

- The Feast reads return `NaN` or empty results: the feature timestamps
  have aged out of the 30 day time to live, which happens if you generated
  the data more than 30 days ago. Rebuild and re-materialise:

  ```bash
  uv run python scripts/seed_data.py
  cd feature_repo && uv run feast materialize-incremental $(date -u +'%Y-%m-%dT%H:%M:%S') && cd ..
  ```

- Docker builds fail or containers get killed for no clear reason: Docker
  has too little memory. Give it at least 6 GB in Docker Desktop settings,
  then restart Docker.

- The `retrain_model` task fails with `command returned a non-zero exit
  code -9`: the training run was killed by the out of memory killer inside
  the container. Train with fewer trees (250 fits comfortably, 300 does
  not on an 8 GB Docker virtual machine), or give Docker more memory.

- `DagNotFound: Dag id retrain_on_drift not found in DagModel` right after
  you created the DAG file: the scheduler registers new files on a scan
  that runs every 30 seconds, while `airflow dags list` reads the files
  directly and shows it immediately. Wait half a minute and trigger again.

- `INFO ... Waiting up to 300 seconds for model version to finish creation`
  when registering a model: this is normal and returns immediately against
  the local server. It is not a hang.

- `WARNING mlflow.utils.environment: Failed to resolve installed pip
  version` when training with tracking on: harmless. MLflow could not pin
  the pip version in the model's environment file; everything else is
  recorded correctly.

- `Downloading artifacts: 100%` progress bars when loading a model: normal
  MLflow output, not an error.

- `curl http://localhost:8000/metrics` prints nothing: the metrics app
  lives at the path with a trailing slash. Use
  `curl -s http://localhost:8000/metrics/ | grep prediction`. Prometheus
  itself follows the redirect and is unaffected.

- `Matplotlib is building the font cache` on the first `feast apply`: a
  one-time message from a Feast dependency. It never appears again.

- `empty cryptography key - values will not be stored encrypted` in Airflow
  logs: expected. The lab sets no Fernet key because it stores no credentials
  in Airflow. See .env.example if you want to set one.

- Start over completely: `docker compose down -v` removes the containers
  and their data volumes. Your files and the MLflow artifacts folder stay.

## Pinned versions

Python 3.11, MLflow 3.15.1, Feast 0.49.0, Evidently 0.5.1, scikit-learn
1.5.2, pandas 2.2.3, numpy 1.26.4, FastAPI 0.115.6, Airflow 2.10.4,
Postgres 16, Redis 7.4, Prometheus v2.55.1, Grafana 11.3.0.
