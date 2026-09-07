# Reference solution for exercise 06, feature stores.
from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64
from feast.value_type import ValueType

district = Entity(
    name="district_id",
    join_keys=["district_id"],
    value_type=ValueType.INT64,
)

source = FileSource(
    path="data/district_features.parquet",
    timestamp_field="event_timestamp",
)

district_features = FeatureView(
    name="district_features",
    entities=[district],
    ttl=timedelta(days=30),
    schema=[
        Field(name="MedInc", dtype=Float64),
        Field(name="HouseAge", dtype=Float64),
        Field(name="Population", dtype=Float64),
    ],
    source=source,
)
