"""
Ames Housing dataset loader — loads real Kaggle CSV.
Selects key numeric features + target (SalePrice).
"""

import os
import pandas as pd


_CSV_PATH = os.path.join(os.path.dirname(__file__), "AmesHousing.csv")

# Numeric columns we keep for the dashboard (avoids huge sparse matrix)
_KEEP_COLS = [
    "Lot Area",
    "Overall Qual",
    "Overall Cond",
    "Year Built",
    "Year Remod/Add",
    "Mas Vnr Area",
    "BsmtFin SF 1",
    "Total Bsmt SF",
    "1st Flr SF",
    "2nd Flr SF",
    "Gr Liv Area",
    "Full Bath",
    "Half Bath",
    "Bedroom AbvGr",
    "Kitchen AbvGr",
    "TotRms AbvGrd",
    "Fireplaces",
    "Garage Cars",
    "Garage Area",
    "Wood Deck SF",
    "Open Porch SF",
    "SalePrice",
]


def get_ames_df() -> pd.DataFrame:
    """Load the real Ames Housing CSV and return a clean numeric DataFrame."""
    df = pd.read_csv(_CSV_PATH)

    # Keep only the columns that exist in the file
    cols = [c for c in _KEEP_COLS if c in df.columns]
    df = df[cols].copy()

    # Fill missing numerics with column median, drop any remaining NaN rows
    for col in df.columns:
        if df[col].dtype in ("float64", "int64", "float32", "int32"):
            df[col] = df[col].fillna(df[col].median())

    df = df.dropna()

    # Clean column names: replace spaces/slashes with underscores
    df.columns = (
        df.columns.str.strip()
        .str.replace(" ", "_")
        .str.replace("/", "_")
    )

    return df
