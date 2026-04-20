"""
Variance thresholding component — normalised variance bar chart + filter logic.
"""

import plotly.graph_objects as go
import pandas as pd
from sklearn.preprocessing import StandardScaler

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="system-ui", color="#333"),
    margin=dict(l=40, r=20, t=40, b=40),
)


def _scaled_variance(numeric_df: pd.DataFrame) -> pd.Series:
    """
    Compute variance after StandardScaler normalisation.
    This ensures features of different scales are compared fairly.
    Then min-max normalise the result to 0–1.
    """
    scaler = StandardScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(numeric_df),
        columns=numeric_df.columns,
        index=numeric_df.index,
    )
    raw_var = scaled.var()

    # Min-max normalise
    v_min, v_max = raw_var.min(), raw_var.max()
    if v_max - v_min == 0:
        return raw_var * 0.0 + 1.0  # all equal → all kept
    return (raw_var - v_min) / (v_max - v_min)


def variance_bar_chart(df: pd.DataFrame, threshold: float = 0.1) -> tuple:
    """
    Return (figure, kept_count, removed_count).
    Variances are computed on standardised data, then min-max normalised to 0–1.
    """
    numeric_df = df.select_dtypes(include="number")
    norm_var = _scaled_variance(numeric_df).sort_values(ascending=True)

    colors = ["#ef4444" if v < threshold else "#0d9488" for v in norm_var]
    kept = sum(1 for v in norm_var if v >= threshold)
    removed = len(norm_var) - kept

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=norm_var.values,
            y=norm_var.index.tolist(),
            orientation="h",
            marker=dict(color=colors, line=dict(color="#fff", width=0.5)),
        )
    )
    # Threshold reference line
    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="#f59e0b",
        line_width=2,
        annotation_text=f"Threshold = {threshold}",
        annotation_position="top right",
        annotation_font=dict(size=11, color="#f59e0b"),
    )
    fig.update_layout(
        title=dict(text="Feature Variance (normalised)", font=dict(size=14)),
        xaxis_title="Normalised Variance",
        yaxis_title="Feature",
        height=max(340, 30 * len(norm_var)),
        **CHART_LAYOUT,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eee", zeroline=False)
    fig.update_yaxes(showgrid=False)
    return fig, kept, removed


def filter_low_variance(df: pd.DataFrame, threshold: float = 0.1) -> pd.DataFrame:
    """Remove features whose normalised variance is below threshold."""
    numeric_df = df.select_dtypes(include="number")
    norm_var = _scaled_variance(numeric_df)
    keep_cols = norm_var[norm_var >= threshold].index.tolist()
    return df[keep_cols]
