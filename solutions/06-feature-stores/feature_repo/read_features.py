# Reference solution for exercise 06, feature stores.
from datetime import datetime

import pandas as pd
from feast import FeatureStore

store = FeatureStore(repo_path="feature_repo")
features = ["district_features:MedInc"]

# Offline: the path training uses. A point-in-time join.
entity_df = pd.DataFrame({"district_id": [3], "event_timestamp": [datetime.now()]})
offline = store.get_historical_features(entity_df=entity_df, features=features).to_df()
offline_value = offline["MedInc"].iloc[0]

# Online: the path inference uses. A single-key lookup.
online = store.get_online_features(features=features, entity_rows=[{"district_id": 3}]).to_dict()
online_value = online["MedInc"][0]

print(f"offline MedInc (district 3): {offline_value:.6f}")
print(f"online  MedInc (district 3): {online_value:.6f}")
print("MATCH" if abs(offline_value - online_value) < 1e-6 else "MISMATCH")
