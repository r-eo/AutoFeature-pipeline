"""
PCA panel — scree plot + PC1 vs PC2 scatter.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="system-ui", color="#333"),
    margin=dict(l=40, r=20, t=40, b=40),
)


def pca_figures(
    df: pd.DataFrame, target_col: str, n_components: int = 5
) -> tuple:
    """
    Return (scree_figure, scatter_figure).
    - scree: bar = explained variance per PC, line = cumulative %.
    - scatter: PC1 vs PC2, colored by target (quartile-binned for continuous,
      raw for discrete ≤10 unique).
    """
    numeric_df = df.select_dtypes(include="number").drop(columns=[target_col], errors="ignore")
    if numeric_df.shape[1] < 2:
        empty = go.Figure().update_layout(**CHART_LAYOUT)
        return empty, empty

    n_components = min(n_components, numeric_df.shape[1], numeric_df.shape[0])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_df)
    pca = PCA(n_components=n_components, random_state=0)
    pcs = pca.fit_transform(X_scaled)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    # ---- Scree plot ----
    pc_labels = [f"PC{i+1}" for i in range(n_components)]
    scree = make_subplots(specs=[[{"secondary_y": True}]])
    scree.add_trace(
        go.Bar(
            x=pc_labels,
            y=explained,
            name="Explained Var",
            marker=dict(color="#4361ee", opacity=0.85),
        ),
        secondary_y=False,
    )
    scree.add_trace(
        go.Scatter(
            x=pc_labels,
            y=cumulative,
            name="Cumulative %",
            mode="lines+markers",
            line=dict(color="#f59e0b", width=2.5),
            marker=dict(size=7),
        ),
        secondary_y=True,
    )
    scree.update_layout(
        title=dict(text="PCA Scree Plot", font=dict(size=14)),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="center", x=0.5),
        **CHART_LAYOUT,
    )
    scree.update_yaxes(title_text="Explained Variance", secondary_y=False, showgrid=True, gridcolor="#eee")
    scree.update_yaxes(title_text="Cumulative %", secondary_y=True, showgrid=False, range=[0, 1.05])
    scree.update_xaxes(showgrid=False)

    # ---- PC1 vs PC2 scatter ----
    target_vals = df[target_col].values if target_col in df.columns else np.zeros(len(df))
    nunique = len(np.unique(target_vals))
    if nunique <= 10:
        color_label = target_vals.astype(str)
        color_map = None
    else:
        quartiles = pd.qcut(target_vals, q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
        color_label = quartiles.astype(str)
        color_map = None

    scatter = go.Figure()
    for label in sorted(set(color_label)):
        mask = color_label == label
        scatter.add_trace(
            go.Scatter(
                x=pcs[mask, 0],
                y=pcs[mask, 1],
                mode="markers",
                name=str(label),
                marker=dict(size=6, opacity=0.75, line=dict(width=0.4, color="#fff")),
            )
        )
    scatter.update_layout(
        title=dict(text="PC1 vs PC2", font=dict(size=14)),
        xaxis_title="PC1",
        yaxis_title="PC2",
        height=380,
        legend=dict(title=target_col, font=dict(size=10)),
        **CHART_LAYOUT,
    )
    scatter.update_xaxes(showgrid=True, gridcolor="#eee", zeroline=False)
    scatter.update_yaxes(showgrid=True, gridcolor="#eee", zeroline=False)

    return scree, scatter
