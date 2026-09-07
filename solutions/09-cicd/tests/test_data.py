# Reference solution for exercise 09, CI/CD.
import pandas as pd

from src.features import FEATURE_NAMES, TARGET


def load():
    return pd.read_csv("data/reference.csv")


def test_expected_columns_present():
    df = load()
    for col in FEATURE_NAMES + [TARGET]:
        assert col in df.columns, f"missing column {col}"


def test_no_nulls_in_features():
    df = load()
    assert df[FEATURE_NAMES].isna().sum().sum() == 0


def test_values_in_plausible_range():
    df = load()
    assert df["MedInc"].between(0, 20).all()
    assert df["HouseAge"].between(0, 100).all()
    assert df[TARGET].between(0, 10).all()
