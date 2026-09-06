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

docker compose up -d

export MLFLOW_TRACKING_URI=http://localhost:5001
```

The first `docker compose up` builds two images and can take around ten
minutes. After that it is cached.

Check that training works:

```bash
uv run python -m src.train
```

You should see an RMSE and a `model.pkl` file.

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

## Troubleshooting

- Port already in use: another program owns the port. `docker ps` shows other
  containers; `lsof -i :5001` (or the port in question) shows host programs.
  Stop whatever owns it and run `docker compose up -d` again.
- A service keeps restarting: read its logs with
  `docker compose logs <service-name> --tail 50`.
- Airflow login page not loading: the webserver takes a minute or two on a
  cold start. Check progress with `docker compose logs airflow-webserver`.
- Start over completely: `docker compose down -v` removes the containers and
  their data volumes. Your files and the MLflow artifacts folder stay.

## Pinned versions

Python 3.11, MLflow 3.15.1, Feast 0.49.0, Evidently 0.5.1, scikit-learn
1.5.2, pandas 2.2.3, numpy 1.26.4, FastAPI 0.115.6, Airflow 2.10.4,
Postgres 16, Redis 7.4, Prometheus v2.55.1, Grafana 11.3.0.
