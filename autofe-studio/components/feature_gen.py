"""
Feature generation component — polynomial, interaction, and log transforms.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


def generate_features(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """
    Generate new features from numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (all columns used; target should be included).
    method : str
        One of "poly", "interaction", "log".

    Returns
    -------
    pd.DataFrame with original + new features.
    """
    numeric_df = df.select_dtypes(include="number").copy()

    if method == "poly":
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
        transformed = poly.fit_transform(numeric_df)
        new_names = poly.get_feature_names_out(numeric_df.columns.tolist())
        result = pd.DataFrame(transformed, columns=new_names, index=df.index)
        return result

    elif method == "interaction":
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
        transformed = poly.fit_transform(numeric_df)
        new_names = poly.get_feature_names_out(numeric_df.columns.tolist())
        result = pd.DataFrame(transformed, columns=new_names, index=df.index)
        return result

    elif method == "log":
        log_df = numeric_df.copy()
        new_cols = {}
        for col in numeric_df.columns:
            min_val = numeric_df[col].min()
            if min_val > 0:
                new_cols[f"log_{col}"] = np.log(numeric_df[col])
            else:
                new_cols[f"log1p_{col}"] = np.log1p(numeric_df[col] - min_val)
        log_additions = pd.DataFrame(new_cols, index=df.index)
        result = pd.concat([numeric_df, log_additions], axis=1)
        return result

    return numeric_df
