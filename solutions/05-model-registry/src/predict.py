# Reference solution for exercise 05, model registry.
import mlflow
import pandas as pd

from src.features import SAMPLE_ROW

model = mlflow.pyfunc.load_model("models:/housing-predictor@production")
prediction = model.predict(pd.DataFrame([SAMPLE_ROW]))
print(f"Prediction: {prediction[0]:.4f}")
