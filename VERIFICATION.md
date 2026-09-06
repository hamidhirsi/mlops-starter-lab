# Exercise verification report

Every exercise page was executed end to end on a clean clone, following each
snippet and command exactly as written, in order. This file records what
happened, every correction the pages need, and every harmless warning a
student will see. Verified on 2026-09-06 on macOS (Apple Silicon) with Docker
Desktop and 8 GB of Docker memory.

## Result per exercise

| Exercise | Result |
| --- | --- |
| 01 Set Up Your Lab | Worked as written |
| 02 Train the Model and Lose the Results | Worked as written |
| 03 Map the Services to the Lifecycle | Worked as written |
| 04 Add Tracking to Your Training Script | Worked, one harmless warning, one snippet worth clarifying |
| 05 Build the Registry Workflow | Worked as written, two harmless messages |
| 06 Define Features Once, Read Them Two Ways | Needed a change: the Entity needs a value_type |
| 07 Find the Limits of One Machine | Worked as written |
| 08 Write the Serving Service | Needed a change: the /metrics curl needs a trailing slash |
| 09 Write the Tests and the Pipeline | Needed a change: one-line imports fail the page's own ruff step |
| 10 Build Drift Detection and a Dashboard | Worked as written (the Grafana dashboard step is manual clicking and was verified by running its three queries) |
| 11 Close the Loop | Needed a change: 300 trees is killed by the out of memory killer inside the container; 250 works |

End state after exercise 11: the DAG ran all four tasks green, the production
alias advanced from version 1 to version 3 (the 250 tree retrain, RMSE 0.5032
against the incumbent 0.5039), the guard correctly refused a worse model, and
the serving API returned a different prediction after restarting on the new
version.

## Page corrections

These are the exact edits to make to the student pages.

### MLOps-06, step 2: give the Entity a value_type

Feast 0.49.0 accepts an Entity without a value_type but prints
`DeprecationWarning: Entity value_type will be mandatory in the next release`
on every apply. Adding it removes the warning.

Original:

```python
from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64

district = Entity(name="district_id", join_keys=["district_id"])
```

Corrected:

```python
from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64
from feast.value_type import ValueType

district = Entity(
    name="district_id",
    join_keys=["district_id"],
    value_type=ValueType.INT64,
)
```

Everything else on the page worked exactly as written: the plain
`registry: data/registry.db` form, `entity_key_serialization_version: 3`
(accepted with no warning), the relative FileSource path resolving correctly
when read from the repository root, and both reads returning the identical
value 5.540644 for district 3.

### MLOps-08, step 6: the metrics path needs a trailing slash

The mounted metrics app answers at `/metrics/`. A request to `/metrics`
returns an empty 307 redirect, so the page's curl prints nothing and the grep
finds nothing. Prometheus follows the redirect on its own and is unaffected;
only the hand-typed curl is.

Original:

```bash
curl -s http://localhost:8000/metrics | grep prediction
```

Corrected:

```bash
curl -s http://localhost:8000/metrics/ | grep prediction
```

Also worth knowing for the page: the `@app.on_event("startup")` deprecation
warning the page braces the student for never actually appears in the
terminal on FastAPI 0.115.6, because Python hides DeprecationWarning from
library code by default.

### MLOps-09, steps 2 and 3: split the imports one per line

The test snippets open with imports on one line. Step 4 of the same page runs
`uv run ruff check .`, which fails on exactly that style (rule E401, multiple
imports on one line), so a student who copies the snippets exactly cannot pass
the page's own lint step.

Original (test_model.py, and the same pattern in test_pipeline.py):

```python
import os, subprocess, sys, joblib, numpy as np, pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from src.features import FEATURE_NAMES, TARGET
```

Corrected:

```python
import os
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from src.features import FEATURE_NAMES, TARGET
```

Corrected (test_pipeline.py):

```python
import os
import subprocess
import sys

import joblib
import pandas as pd

from src.features import FEATURE_NAMES, SAMPLE_ROW
```

Everything else on the page held: all five tests pass with
`MLFLOW_TRACKING_URI` set in the shell (the tests strip it themselves and the
model.pkl fallback fires), a 50 tree model scores RMSE 0.5064 against the
0.60 threshold, changing the threshold to 0.30 makes the gate fail as
intended, and both action tags (`actions/checkout@v4`,
`astral-sh/setup-uv@v3`) exist.

### MLOps-11, step 3: retrain with 250 trees, not 300

`--n-estimators 300` inside the Airflow container is killed by the out of
memory killer on an 8 GB Docker virtual machine: the task fails with
`command returned a non-zero exit code -9` after about 30 seconds. 250 trees
fits comfortably and still beats the 200 tree incumbent (RMSE 0.5032 against
0.5039), so the guard promotes it and the version still advances on screen.

Original:

```python
    retrain = BashOperator(task_id="retrain_model",
                           bash_command="cd /opt/project && python -m src.train --n-estimators 300")
```

Corrected:

```python
    retrain = BashOperator(task_id="retrain_model",
                           bash_command="cd /opt/project && python -m src.train --n-estimators 250")
```

The "What you should see" paragraph on the page should say 250 instead of
300 in the same breath ("The pipeline retrained with 250 trees, which beats
the 200-tree run").

### MLOps-11, step 4: mention the registration delay

`airflow dags list` reads the DAG files directly and shows a new DAG at once,
but trigger and unpause read the scheduler's database, which only learns
about brand new files on a periodic scan. Following the page quickly produced
`DagNotFound: Dag id retrain_on_drift not found in DagModel` on the trigger.
The starter now sets the scan interval to 30 seconds, so one sentence on the
page is enough. Suggested addition after the trigger command:

> If the trigger says the DAG is not found, wait half a minute and run it
> again: the scheduler picks up brand new files on its next scan.

### MLOps-04, steps 2 and 3: one clarification worth making (optional)

Step 2 shows the training code inside the `with mlflow.start_run():` block,
and step 3 then wraps "the tracking block" in an environment check with the
model.pkl fallback in the else branch. Read together it is ambiguous whether
training runs inside or outside the tracking block. The verified solution
trains once, unconditionally, and only the logging is conditional, which
avoids duplicating the training code:

```python
    # ... training and RMSE code, unchanged ...

    if os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_experiment("housing")

        with mlflow.start_run():
            mlflow.log_param("n_estimators", args.n_estimators)
            mlflow.log_param("max_depth", args.max_depth)
            mlflow.log_metric("rmse", rmse)

            signature = infer_signature(X_train, model.predict(X_train.iloc[:5]))
            mlflow.sklearn.log_model(
                model,
                name="model",
                signature=signature,
                input_example=X_train.iloc[:1],
            )
    else:
        joblib.dump(model, "model.pkl")
        print("Model saved to model.pkl")
```

Both readings work; this one is what the solutions use and what the tests in
exercise 09 rely on.

## Harmless warnings a student will see

Each of these is normal. All are also listed in the README troubleshooting
section so students can look them up by the exact text.

- `WARNING mlflow.utils.environment: Failed to resolve installed pip version`
  on every tracked training run: MLflow could not pin pip's version in the
  model's environment file; the run, metric, and model are all recorded
  correctly.
- `INFO mlflow.store.model_registry.abstract_store: Waiting up to 300
  seconds for model version to finish creation` on every registration: it
  returns immediately against the local server; confirmed by timing.
- `Downloading artifacts: 100%|...|` progress bars whenever a model is loaded
  from the registry: normal MLflow output.
- `Matplotlib is building the font cache; this may take a moment` on the
  first `feast apply` only: a one-time message from a Feast dependency.
- The materialize progress bar (`29/29 [00:00...]`) during
  `feast materialize-incremental`: normal Feast output.

## Changes made to the starter (main)

- `scripts/check_setup.py`: the setup diagnostic the pages reference. Checks
  Python, uv, Docker and its memory, every port, every service's health, the
  three data files, and MLflow reachability, with a fix sentence per failure.
- `scripts/check.py`: the per-exercise checker the pages reference, sections
  4 to 11. Read-only.
- `.env.example` and a `user: "${AIRFLOW_UID:-50000}:0"` line on the shared
  Airflow definition: on Linux, files written by DAG tasks into the mounted
  repository stay owned by the student. Verified the image still works as an
  arbitrary uid (tested uid 12345: Airflow starts and the ML stack imports).
- README setup: `docker compose up -d --wait` (verified it returns only when
  every service is healthy, and that the one-shot airflow-init exiting 0 is
  handled), the Linux `AIRFLOW_UID` line, and a pointer to check_setup.py.
- `AIRFLOW__SCHEDULER__DAG_DIR_LIST_INTERVAL: "30"` on the shared Airflow
  definition: brand new DAG files become triggerable within about 30 seconds
  instead of up to 5 minutes.
- `checkpoint.pkl` added to .gitignore: exercise 07 creates it and it is
  290 MB.
- README troubleshooting: rewritten to cover every problem and warning met
  during this verification, each with the exact message, the cause, and the
  fix.

## Versions verified against

Python 3.11.15, uv managed. MLflow 3.15.1, Feast 0.49.0, Evidently 0.5.1,
scikit-learn 1.5.2, pandas 2.2.3, numpy 1.26.4, FastAPI 0.115.6, uvicorn
0.34.0, pydantic 2.10.6, prometheus-client 0.21.1, pytest 8.3.4, ruff 0.9.6,
pyarrow 17.0.0. Services: Airflow 2.10.4 (Python 3.11 image), Postgres 16,
Redis 7.4, Prometheus v2.55.1, Grafana 11.3.0, MLflow server 3.15.1 built on
python:3.11-slim with cryptography 46.0.3. Docker Compose v2.32.4, Docker
Desktop on macOS, 8 GB memory allocated to Docker.
