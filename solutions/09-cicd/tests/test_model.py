# Reference solution for exercise 09, CI/CD.
import os
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from src.features import FEATURE_NAMES, TARGET

RMSE_THRESHOLD = 0.60


def test_model_beats_threshold():
    env = {k: v for k, v in os.environ.items() if k != "MLFLOW_TRACKING_URI"}
    subprocess.run([sys.executable, "-m", "src.train", "--n-estimators", "50"],
                   check=True, env=env)
    model = joblib.load("model.pkl")
    df = pd.read_csv("data/reference.csv")
    _, X_test, _, y_test = train_test_split(df[FEATURE_NAMES], df[TARGET],
                                            test_size=0.2, random_state=42)
    rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
    assert rmse < RMSE_THRESHOLD, f"RMSE {rmse:.4f} is worse than {RMSE_THRESHOLD}"
