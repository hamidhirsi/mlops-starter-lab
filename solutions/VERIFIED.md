# Solution verification record

Every solution in this folder was copied into its real location on a running
lab and proven against its own check script, in exercise order, on a stack
started fresh with `docker compose down -v` first. Verified on 2026-09-07 on
macOS (Apple Silicon), Docker Desktop with 8 GB of memory.

| Exercise | Proof | check.py |
| --- | --- | --- |
| 04 tracking | four tracked runs (RMSE 0.5122, 0.5064, 0.5057, 0.5039), each with params, metric, and model | pass |
| 05 registry | registered v1, production alias set, predict by alias returned 3.0774 | pass |
| 06 feature store | apply, materialize (29 districts), offline and online reads both 5.540644, MATCH | pass |
| 07 checkpointing | four checkpoints written; checkpoint file read mid run held exactly the partial tree count | pass |
| 08 serving | /ready 200, predictions rise with income, bad type rejected with 422, metrics exposed; the Dockerfile builds and the container serves the registry model | pass |
| 09 CI | ruff clean, five tests pass, threshold 0.30 makes the gate fail on purpose, workflow parses and its action tags exist | pass |
| 10 monitoring | drift share 0.25 with exactly MedInc and HouseAge flagged; all three Prometheus queries return data; the dashboard JSON provisions in Grafana with all four panels | pass |
| 11 pipeline | full run green; a second run correctly kept the incumbent ("Keeping current model (RMSE 0.5032); best run 0.5032 is not better"); /reload swapped serving between v1 and v2 with no restart | pass |

Also verified as part of this run:

- The student comparison flow: a matching file diffs down to only the header
  line, a broken file fails its check with the real error and the diff
  pinpoints the line, and a correct but differently written file passes its
  check while diffing noisily. The check script is the judge of correctness;
  the diff is for comparison.
- Retraining with 300 trees inside the Airflow container is killed by the out
  of memory killer on an 8 GB Docker virtual machine. The DAG uses 250, which
  fits and still beats the 200 tree incumbent (0.5032 against 0.5039).
- The DAG sets max_active_runs to 1: unpausing a daily DAG starts a catch up
  run for the missed interval, and the exercise triggers a manual run seconds
  later. Without the limit the two retrainings ran at once and one was killed
  for memory.
- The Airflow services run with an empty Fernet key (the lab stores no
  credentials in Airflow); the stack comes up healthy with it.

Versions: Python 3.11, MLflow 3.15.1, Feast 0.49.0, Evidently 0.5.1,
scikit-learn 1.5.2, pandas 2.2.3, numpy 1.26.4, FastAPI 0.115.6, uvicorn
0.34.0, prometheus-client 0.21.1, Airflow 2.10.4, Postgres 16, Redis 7.4,
Prometheus v2.55.1, Grafana 11.3.0, Docker Compose v2.32.4.
