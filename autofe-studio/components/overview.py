"""
Overview component — dataset summary metric cards + target histogram.
"""

import plotly.graph_objects as go
import pandas as pd

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="system-ui", color="#333"),
    margin=dict(l=40, r=20, t=40, b=40),
)


def compute_stats(df: pd.DataFrame) -> dict:
    """Return quick summary statistics for the DataFrame."""
    total_rows = len(df)
    total_cols = len(df.columns)
    numeric_cols = len(df.select_dtypes(include="number").columns)
    missing_pct = round(df.isnull().sum().sum() / (total_rows * total_cols) * 100, 2)
    return {
        "rows": total_rows,
        "cols": total_cols,
        "numeric": numeric_cols,
        "missing_pct": missing_pct,
    }


def target_histogram(df: pd.DataFrame, target_col: str) -> go.Figure:
    """Return a Plotly histogram figure for the target column."""
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=df[target_col],
            marker=dict(
                color="#4361ee",
                line=dict(color="#2e47b0", width=0.8),
            ),
            opacity=0.88,
            nbinsx=30,
            name=target_col,
        )
    )
    fig.update_layout(
        title=dict(text=f"Distribution of {target_col}", font=dict(size=14)),
        xaxis_title=target_col,
        yaxis_title="Count",
        bargap=0.05,
        **CHART_LAYOUT,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eee", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#eee", zeroline=False)
    return fig
