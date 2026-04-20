"""
SHAP-style feature importance panel — uses sklearn permutation_importance.
"""

import plotly.graph_objects as go
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="system-ui", color="#333"),
    margin=dict(l=40, r=20, t=40, b=40),
)


def importance_chart(df: pd.DataFrame, target_col: str, top_n: int = 15) -> go.Figure:
    """
    Train a RandomForest, compute permutation importance, return a horizontal bar chart.
    Uses classifier for discrete targets (≤10 unique) and regressor otherwise.
    """
    numeric_df = df.select_dtypes(include="number").dropna(axis=1).copy()
    if target_col not in numeric_df.columns:
        return go.Figure().update_layout(**CHART_LAYOUT)

    y = numeric_df[target_col]
    X = numeric_df.drop(columns=[target_col])

    if X.empty or len(X) < 10:
        return go.Figure().update_layout(**CHART_LAYOUT)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    nunique = y.nunique()
    if nunique <= 10:
        model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)
    else:
        model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)

    model.fit(X_scaled, y)

    perm_imp = permutation_importance(
        model, X_scaled, y, n_repeats=5, random_state=42, n_jobs=1
    )

    imp_series = pd.Series(perm_imp.importances_mean, index=X.columns)
    imp_series = imp_series.sort_values(ascending=True).tail(top_n)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=imp_series.values,
            y=imp_series.index.tolist(),
            orientation="h",
            marker=dict(
                color="#0d9488",
                line=dict(color="#fff", width=0.5),
            ),
        )
    )
    fig.update_layout(
        title=dict(text=f"Top {top_n} Feature Importance (Permutation)", font=dict(size=14)),
        xaxis_title="Mean Importance",
        yaxis_title="Feature",
        height=max(380, 28 * top_n),
        **CHART_LAYOUT,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eee", zeroline=False)
    fig.update_yaxes(showgrid=False)
    return fig
