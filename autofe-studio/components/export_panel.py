"""
Export panel — assemble selected transformations into a downloadable CSV.
"""

import io
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from components.variance import filter_low_variance


def build_export_df(
    df: pd.DataFrame,
    selections: list,
    variance_threshold: float = 0.1,
) -> str:
    """
    Build a combined DataFrame based on user checklist selections
    and return it as a CSV string.

    selections is a list of strings from:
        ["original", "polynomial", "pca", "low_variance"]
    """
    numeric_df = df.select_dtypes(include="number").copy()
    parts = []

    if "original" in selections:
        parts.append(numeric_df)

    if "polynomial" in selections:
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
        transformed = poly.fit_transform(numeric_df)
        names = poly.get_feature_names_out(numeric_df.columns.tolist())
        # Only keep the new columns (skip original columns)
        new_mask = [n for n in names if n not in numeric_df.columns]
        poly_df = pd.DataFrame(transformed, columns=names, index=df.index)
        parts.append(poly_df[new_mask])

    if "pca" in selections:
        n_comp = min(5, numeric_df.shape[1], numeric_df.shape[0])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(numeric_df)
        pca = PCA(n_components=n_comp, random_state=0)
        pcs = pca.fit_transform(X_scaled)
        pc_cols = [f"PC{i+1}" for i in range(n_comp)]
        parts.append(pd.DataFrame(pcs, columns=pc_cols, index=df.index))

    if "low_variance" in selections:
        filtered = filter_low_variance(numeric_df, threshold=variance_threshold)
        # Rename to avoid collision when "original" is also selected
        filtered = filtered.rename(columns={c: f"lv_{c}" for c in filtered.columns})
        parts.append(filtered)

    if not parts:
        parts.append(numeric_df)

    combined = pd.concat(parts, axis=1)
    # Remove any duplicate columns
    combined = combined.loc[:, ~combined.columns.duplicated()]

    buf = io.StringIO()
    combined.to_csv(buf, index=False)
    return buf.getvalue()
