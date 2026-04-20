"""
Correlation component — Pearson correlation heatmap + top-5 pairs table.
"""

import itertools
import plotly.graph_objects as go
import pandas as pd

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="system-ui", color="#333"),
    margin=dict(l=40, r=20, t=40, b=40),
)


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Return an imshow-style heatmap of Pearson correlations."""
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.columns.tolist(),
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            text=corr.values.round(2),
            texttemplate="%{text}",
            textfont=dict(size=10),
            colorbar=dict(title="r", thickness=14, len=0.6),
        )
    )
    fig.update_layout(
        title=dict(text="Pearson Correlation Matrix", font=dict(size=14)),
        xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=10), autorange="reversed"),
        height=520,
        **CHART_LAYOUT,
    )
    return fig


def top_correlated_pairs(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Return a DataFrame of the top-N most correlated feature pairs."""
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr().abs()

    pairs = []
    for c1, c2 in itertools.combinations(corr.columns, 2):
        pairs.append({"Feature A": c1, "Feature B": c2, "Correlation": round(corr.loc[c1, c2], 4)})

    pairs_df = pd.DataFrame(pairs).sort_values("Correlation", ascending=False).head(top_n).reset_index(drop=True)
    return pairs_df
