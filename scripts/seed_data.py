"""Generate the datasets the exercises start from.

Writes two things:
- data/reference.csv: the California housing dataset, 8 features plus the
  MedHouseVal target. This is your training data.
- feature_repo/data/district_features.parquet: per-district aggregate features
  with an event_timestamp column, shaped the way a feature store expects its
  source data. You will point Feast at this file in the feature store exercise.
"""
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing


def main():
    # These folders are gitignored except for a placeholder, so they can be
    # empty on a fresh clone.
    os.makedirs("data", exist_ok=True)
    os.makedirs("feature_repo/data", exist_ok=True)

    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df["MedHouseVal"] = housing.target

    df.to_csv("data/reference.csv", index=False)
    print(f"Wrote data/reference.csv with {len(df)} rows")

    # Build a district id by binning the map into an 8 by 6 grid: 48 possible
    # districts. Some grid cells contain no houses, so not every id appears.
    lat_bins = pd.cut(df["Latitude"], bins=8, labels=False)
    lon_bins = pd.cut(df["Longitude"], bins=6, labels=False)
    df["district_id"] = lat_bins * 6 + lon_bins

    district_features = (
        df.groupby("district_id")
        .agg({"MedInc": "mean", "HouseAge": "mean", "Population": "mean"})
        .reset_index()
    )

    # A feature store needs to know WHEN each feature value became true.
    # Give every district a timestamp somewhere in the last 7 days.
    rng = np.random.default_rng(42)
    now = datetime.now()
    district_features["event_timestamp"] = [
        now - timedelta(days=float(rng.uniform(0, 7)))
        for _ in range(len(district_features))
    ]

    district_features.to_parquet(
        "feature_repo/data/district_features.parquet", index=False
    )
    print(
        f"Wrote feature_repo/data/district_features.parquet "
        f"with {len(district_features)} districts"
    )


if __name__ == "__main__":
    main()
