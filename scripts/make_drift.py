"""Create a drifted copy of the reference data.

Writes data/drifted.csv: the same rows as data/reference.csv, except that
median income is 60 percent higher and every house is 15 years older. Every
other column is unchanged. You will use this file in the monitoring exercise
to make a drift detector fire on purpose.
"""
import pandas as pd


def main():
    reference = pd.read_csv("data/reference.csv")

    drifted = reference.copy()
    drifted["MedInc"] = drifted["MedInc"] * 1.6
    drifted["HouseAge"] = drifted["HouseAge"] + 15

    drifted.to_csv("data/drifted.csv", index=False)
    print(f"Wrote data/drifted.csv with {len(drifted)} rows")


if __name__ == "__main__":
    main()
