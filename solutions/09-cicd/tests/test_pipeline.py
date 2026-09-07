# Reference solution for exercise 09, CI/CD.
import os
import subprocess
import sys

import joblib
import pandas as pd

from src.features import FEATURE_NAMES, SAMPLE_ROW


def test_training_produces_usable_model():
    env = {k: v for k, v in os.environ.items() if k != "MLFLOW_TRACKING_URI"}
    result = subprocess.run([sys.executable, "-m", "src.train", "--n-estimators", "10"],
                            capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    assert "RMSE:" in result.stdout
    model = joblib.load("model.pkl")
    prediction = model.predict(pd.DataFrame([SAMPLE_ROW])[FEATURE_NAMES])[0]
    assert 0 < prediction < 10
