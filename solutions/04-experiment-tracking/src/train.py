# Reference solution for exercise 04, experiment tracking.
"""Train a random forest on the California housing data.

Run it from the repository root:

    uv run python -m src.train --n-estimators 100

When MLFLOW_TRACKING_URI is set, every run is logged to the tracking server:
parameters, the RMSE, and the model itself. Without it, the model is saved to
model.pkl, which is what the tests rely on.
"""
import argparse
import os

os.environ.setdefault("MLFLOW_UV_AUTO_DETECT", "false")  # record only what the model uses

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from src.features import FEATURE_NAMES, TARGET


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=50)
    parser.add_argument("--max-depth", type=int, default=None)
    args = parser.parse_args()

    df = pd.read_csv("data/reference.csv")
    X = df[FEATURE_NAMES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))

    print(f"RMSE: {rmse:.4f}")

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


if __name__ == "__main__":
    main()
