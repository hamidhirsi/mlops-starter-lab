"""The feature contract: the single source of truth for feature names.

Training, serving, and any script that builds model input should import these
names instead of keeping their own copy. When two parts of a system each
maintain their own feature list, they eventually disagree, and the model
quietly receives columns in the wrong order. Importing from one place makes
that mistake impossible.
"""

FEATURE_NAMES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]

TARGET = "MedHouseVal"

# One plausible row of input, useful for smoke tests and example requests.
SAMPLE_ROW = {
    "MedInc": 5.0,
    "HouseAge": 30.0,
    "AveRooms": 6.0,
    "AveBedrms": 1.1,
    "Population": 800.0,
    "AveOccup": 2.5,
    "Latitude": 34.0,
    "Longitude": -118.0,
}
