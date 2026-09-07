# Reference solution for exercise 10, monitoring.
import json

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from src.features import FEATURE_NAMES

reference = pd.read_csv("data/reference.csv")[FEATURE_NAMES]
current = pd.read_csv("data/drifted.csv")[FEATURE_NAMES]

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference, current_data=current)
report.save_html("drift_report.html")

result = report.as_dict()["metrics"][0]["result"]
summary = {
    "share_of_drifted_columns": result["share_of_drifted_columns"],
    "dataset_drift": result["dataset_drift"],
}
json.dump(summary, open("drift_summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
