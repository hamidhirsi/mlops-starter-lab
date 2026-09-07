# Reference solution for exercise 11, production (adds the quality guard).
import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "housing-predictor"
client = MlflowClient()

# 1. Find the run with the lowest RMSE
experiment = client.get_experiment_by_name("housing")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.rmse ASC"],
    max_results=1,
)
best = runs[0]
print(f"Best run: {best.info.run_id} RMSE: {best.data.metrics['rmse']:.4f}")

# Only promote when the best run actually beats the model in production.
# A pipeline must be allowed to conclude that the current model should stay.
try:
    current = client.get_model_version_by_alias(MODEL_NAME, "production")
    current_rmse = client.get_run(current.run_id).data.metrics["rmse"]
except Exception:
    current_rmse = float("inf")   # nothing in production yet

best_rmse = best.data.metrics["rmse"]
if best_rmse >= current_rmse:
    print(f"Keeping current model (RMSE {current_rmse:.4f}); best run {best_rmse:.4f} is not better")
    raise SystemExit(0)

# 2. Find the model that run logged. In MLflow 3 a logged model is its own
#    entity, so look it up by the run that produced it.
logged = client.search_logged_models(
    experiment_ids=[experiment.experiment_id],
    filter_string=f"source_run_id='{best.info.run_id}'",
)
model_uri = f"models:/{logged[0].model_id}"

# 3. Register it (creates the registered model on first use)
version = mlflow.register_model(model_uri, MODEL_NAME)
print(f"Registered version {version.version}")

# 4. Point the production alias at it
client.set_registered_model_alias(MODEL_NAME, "production", version.version)
print(f"production -> version {version.version}")
