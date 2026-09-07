# Reference solutions

One reference solution per exercise, snapshotted at the end of that exercise.
Each subfolder contains only what you have learned by that point in the
module. The exercise 04 training script has tracking but no checkpointing,
because checkpointing arrives in exercise 07. You never see code from ahead
of where you are.

## How to use this folder

Attempt the exercise first: that is where the learning happens. Then run the
exercise's check:

```bash
uv run python scripts/check.py 5
```

When it passes, compare your file against the reference. Comparing after a
passing check confirms the shape of what you built and often shows another
reasonable way to do the same thing. If you have been stuck for more than
twenty minutes, comparing early is a good use of this folder too. Reading
the solution before attempting teaches very little.

## How to compare

The subfolders mirror the repository layout, so every comparison is one
command, your file first, the reference second:

```bash
diff src/register.py solutions/05-model-registry/src/register.py
```

Every solution file starts with a one line reference header, so even a
perfect match shows exactly one difference: that header. Anything beyond the
header line is a real difference. Add `-u` for a more readable view:

```bash
diff -u src/train.py solutions/04-experiment-tracking/src/train.py
```

One important thing about differences: the check script is the judge of
whether your work is correct, not the diff. Your variable names, comments,
and ordering will differ from the reference, and that is fine. A difference
is a comparison point, not a mistake. What matters is whether the check
passes and whether you can explain why both versions work.

## What each subfolder contains

- `04-experiment-tracking/`: train.py with MLflow tracking and the model.pkl fallback
- `05-model-registry/`: register.py (find best run, register, set the alias) and predict.py (load by alias)
- `06-feature-stores/`: the Feast definitions and the two-path read script
- `07-training-infrastructure/`: train.py with the checkpoint option added
- `08-inference-serving/`: the FastAPI service, plus the Dockerfile from the go further step
- `09-cicd/`: the three test files and the GitHub Actions workflow
- `10-monitoring/`: the drift check, plus the four panel dashboard JSON from the go further step
- `11-production/`: register.py with the quality guard, the retraining DAG, and serve.py with the /reload endpoint from the go further step
