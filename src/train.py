"""Train a random forest on the California housing data.

This is a plain training script on purpose. It knows nothing about experiment
tracking, registries, or serving: you will add those layers yourself in the
course exercises. Run it from the repository root:

    uv run python -m src.train --n-estimators 100
"""
import argparse

import joblib
import numpy as np
import pandas as pd
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

    joblib.dump(model, "model.pkl")
    print("Model saved to model.pkl")


if __name__ == "__main__":
    main()
