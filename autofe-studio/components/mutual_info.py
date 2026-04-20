"""
Mutual Information component — MI scores bar chart.
"""

import plotly.graph_objects as go
import pandas as pd
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="system-ui", color="#333"),
    margin=dict(l=40, r=20, t=40, b=40),
)


def mutual_info_chart(df: pd.DataFrame, target_col: str) -> go.Figure:
    """
    Compute MI scores of every numeric feature vs target_col.
    Uses regression MI for continuous targets, classification MI for discrete.
    Returns a horizontal bar chart sorted descending.
    """
    numeric_df = df.select_dtypes(include="number").copy()
    if target_col not in numeric_df.columns:
        # Return empty figure
        return go.Figure().update_layout(**CHART_LAYOUT)

    y = numeric_df[target_col]
    X = numeric_df.drop(columns=[target_col])

    if X.empty:
        return go.Figure().update_layout(**CHART_LAYOUT)

    # Decide MI function based on target cardinality
    nunique = y.nunique()
    if nunique <= 10:
        mi = mutual_info_classif(X, y, random_state=0)
    else:
        mi = mutual_info_regression(X, y, random_state=0)

    mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=True)

    # Sequential teal colour mapping
    norm_scores = (mi_series - mi_series.min())
    if norm_scores.max() > 0:
        norm_scores = norm_scores / norm_scores.max()
    colors = [
        f"rgba({int(13 + (1 - v) * 100)}, {int(148 + (1 - v) * 60)}, {int(136 + (1 - v) * 40)}, 0.92)"
        for v in norm_scores
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=mi_series.values,
            y=mi_series.index.tolist(),
            orientation="h",
            marker=dict(color=colors, line=dict(color="#fff", width=0.5)),
        )
    )
    fig.update_layout(
        title=dict(text=f"Mutual Information vs {target_col}", font=dict(size=14)),
        xaxis_title="MI Score",
        yaxis_title="Feature",
        height=max(340, 30 * len(mi_series)),
        **CHART_LAYOUT,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eee", zeroline=False)
    fig.update_yaxes(showgrid=False)
    return fig
