"""
Credit Card Fraud dataset loader — loads real Kaggle CSV.
Engineers numeric features from categorical columns for meaningful analysis.
"""

import os
import numpy as np
import pandas as pd


_CSV_PATH = os.path.join(os.path.dirname(__file__), "credit_card_fraud.csv")


def get_fraud_df() -> pd.DataFrame:
    """Load the real Credit Card Fraud CSV and engineer numeric features."""
    df = pd.read_csv(_CSV_PATH)

    result = pd.DataFrame(index=df.index)

    # --- Direct numeric columns ---
    if "Transaction Amount" in df.columns:
        result["Amount"] = pd.to_numeric(df["Transaction Amount"], errors="coerce")

    if "Merchant Category Code (MCC)" in df.columns:
        result["MCC"] = pd.to_numeric(df["Merchant Category Code (MCC)"], errors="coerce")

    if "Transaction Response Code" in df.columns:
        result["ResponseCode"] = pd.to_numeric(df["Transaction Response Code"], errors="coerce")

    # --- Engineered features from datetime ---
    if "Transaction Date and Time" in df.columns:
        dt = pd.to_datetime(df["Transaction Date and Time"], errors="coerce")
        result["Hour"] = dt.dt.hour
        result["DayOfWeek"] = dt.dt.dayofweek
        result["Month"] = dt.dt.month
        result["IsWeekend"] = (dt.dt.dayofweek >= 5).astype(int)
        result["IsNight"] = ((dt.dt.hour >= 22) | (dt.dt.hour <= 5)).astype(int)

    # --- Engineered features from categorical ---
    if "Card Type" in df.columns:
        card_dummies = pd.get_dummies(df["Card Type"], prefix="Card", dtype=int)
        result = pd.concat([result, card_dummies], axis=1)

    if "Transaction Source" in df.columns:
        source_dummies = pd.get_dummies(df["Transaction Source"], prefix="Source", dtype=int)
        result = pd.concat([result, source_dummies], axis=1)

    if "Device Information" in df.columns:
        device_dummies = pd.get_dummies(df["Device Information"], prefix="Device", dtype=int)
        result = pd.concat([result, device_dummies], axis=1)

    if "Transaction Currency" in df.columns:
        currency_dummies = pd.get_dummies(df["Transaction Currency"], prefix="Currency", dtype=int)
        result = pd.concat([result, currency_dummies], axis=1)

    if "Previous Transactions" in df.columns:
        prev_map = {"None": 0, "1": 1, "2": 2, "3 or more": 3}
        result["PrevTxnCount"] = df["Previous Transactions"].map(prev_map).fillna(0).astype(int)

    # --- Log-transformed amount ---
    if "Amount" in result.columns:
        result["LogAmount"] = np.log1p(result["Amount"])

    # --- Target: Fraud label ---
    if "Fraud Flag or Label" in df.columns:
        result["Class"] = pd.to_numeric(df["Fraud Flag or Label"], errors="coerce")

    # Drop rows with NaN and reset
    result = result.dropna()

    # Sample for performance
    if len(result) > 5000:
        result = result.sample(n=5000, random_state=42).reset_index(drop=True)

    # Clean column names
    result.columns = (
        result.columns.str.strip()
        .str.replace(" ", "_")
        .str.replace("/", "_")
    )

    return result
