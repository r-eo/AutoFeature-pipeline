"""
AutoFE Studio — Automated Feature Engineering Dashboard
========================================================
Entry point. Run with:  python app.py
Deploy with:            gunicorn app:server
"""

# ─── Imports ────────────────────────────────────────────────────────────────────
import io
import json
import dash
from dash import dcc, html, dash_table, callback_context, no_update
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go

# Local modules
from data.ames_housing import get_ames_df
from data.credit_fraud import get_fraud_df
from components.overview import compute_stats, target_histogram
from components.correlation import correlation_heatmap, top_correlated_pairs
from components.variance import variance_bar_chart
from components.mutual_info import mutual_info_chart
from components.pca_panel import pca_figures
from components.feature_gen import generate_features
from components.shap_panel import importance_chart
from components.export_panel import build_export_df

# ─── Preload datasets ──────────────────────────────────────────────────────────
print("⏳ Loading datasets...")
DATASETS = {
    "ames": get_ames_df(),
    "fraud": get_fraud_df(),
}
TARGET_MAP = {"ames": "SalePrice", "fraud": "Class"}
LABEL_MAP = {"ames": "Ames Housing", "fraud": "Credit Card Fraud"}

# ─── Pre-compute ALL expensive results at startup ──────────────────────────────
print("⏳ Pre-computing analysis cache (this runs once)...")
CACHE = {}
for _key, _df in DATASETS.items():
    _target = TARGET_MAP[_key]
    _features_df = _df.drop(columns=[_target], errors="ignore")
    CACHE[_key] = {
        "stats": compute_stats(_df),
        "histogram": target_histogram(_df, _target),
        "corr_heatmap": correlation_heatmap(_df),
        "corr_pairs": top_correlated_pairs(_df),
        "mi_chart": mutual_info_chart(_df, _target),
        "pca_5": pca_figures(_df, _target, 5),
        "importance": importance_chart(_df, _target, top_n=15),
    }
print("✅ Cache ready — dashboard will load instantly!")

# ─── App initialisation ────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
    title="AutoFE Studio",
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
        {"name": "description", "content": "Automated Feature Engineering Dashboard"},
    ],
)
server = app.server  # Required for gunicorn


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def _get_df(dataset_key: str) -> pd.DataFrame:
    """Load DataFrame from in-memory DATASETS dict.
    Much more reliable than serialising/deserialising huge JSON strings."""
    return DATASETS.get(dataset_key, DATASETS["ames"])


def _empty_fig(msg: str = "No data") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        annotations=[dict(text=msg, showarrow=False, font=dict(size=14, color="#999"))],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  LAYOUT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def _metric_card(card_id: str, label: str, value: str = "—") -> dbc.Col:
    return dbc.Col(
        html.Div(
            [
                html.Div(value, className="metric-value", id=card_id),
                html.Div(label, className="metric-label"),
            ],
            className="metric-card",
        ),
        md=3,
        sm=6,
        xs=6,
        className="mb-3",
    )


# ─── Navbar ─────────────────────────────────────────────────────────────────────
navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                dbc.Row(
                    dbc.Col(
                        html.Div([
                            html.Span("⚡ AutoFE Studio", style={"fontWeight": "700", "fontSize": "1.2rem"}),
                            html.Small("Automated Feature Engineering", style={"display": "block", "opacity": "0.6", "fontSize": "0.72rem", "marginTop": "-2px"}),
                        ])
                    ),
                    align="center",
                ),
                href="#",
                style={"textDecoration": "none", "color": "white"},
            ),
            dbc.Nav(
                [
                    dbc.NavItem(
                        dbc.NavLink(
                            html.I(className="bi bi-github"),
                            href="#",
                            style={"fontSize": "1.2rem"},
                        )
                    ),
                    dbc.NavItem(
                        dbc.Switch(
                            id="dark-mode-toggle",
                            label="🌙",
                            value=False,
                            className="ms-3 mt-1",
                            style={"color": "white"},
                        )
                    ),
                ],
                className="ms-auto",
                navbar=True,
            ),
        ],
        fluid=True,
    ),
    className="navbar-custom",
    dark=True,
    sticky="top",
)

# ─── Sidebar ────────────────────────────────────────────────────────────────────
sidebar = html.Div(
    [
        html.H6("Dataset", className="mb-2", style={"fontWeight": "600", "color": "#1a1a2e"}),
        dbc.RadioItems(
            id="dataset-selector",
            options=[
                {"label": "🏠  Ames Housing", "value": "ames"},
                {"label": "💳  Credit Card Fraud", "value": "fraud"},
            ],
            value="ames",
            className="mb-3",
            inputClassName="me-2",
        ),
        html.Hr(className="section-divider"),
        dbc.Card(
            dbc.CardBody(
                [
                    html.H6("Quick Stats", className="mb-2", style={"fontWeight": "600", "fontSize": "0.85rem"}),
                    html.Div(id="sidebar-stats", children=[
                        html.Div("Loading…", style={"color": "#999", "fontSize": "0.8rem"}),
                    ]),
                ]
            ),
            className="mb-3",
        ),
        html.Hr(className="section-divider"),
        html.H6("Sections", className="mb-2", style={"fontWeight": "600", "color": "#1a1a2e", "fontSize": "0.85rem"}),
        dbc.Nav(
            [
                dbc.NavLink("📊 Overview", href="#overview", external_link=True),
                dbc.NavLink("🔗 Correlation", href="#correlation", external_link=True),
                dbc.NavLink("📉 Variance", href="#variance", external_link=True),
                dbc.NavLink("🎯 Mutual Info", href="#mutual-info", external_link=True),
                dbc.NavLink("🧬 PCA", href="#pca", external_link=True),
                dbc.NavLink("⚙️ Features", href="#features", external_link=True),
                dbc.NavLink("🏆 Importance", href="#shap", external_link=True),
                dbc.NavLink("📥 Export", href="#export", external_link=True),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    className="sidebar",
)

# ─── Section builders ───────────────────────────────────────────────────────────

section_overview = html.Div(
    id="overview",
    children=[
        dbc.Card(
            [
                html.H5("📊 Dataset Overview"),
                html.Hr(className="section-divider"),
                dbc.Row(
                    [
                        _metric_card("stat-rows", "Total Rows"),
                        _metric_card("stat-cols", "Total Features"),
                        _metric_card("stat-numeric", "Numeric Features"),
                        _metric_card("stat-missing", "Missing Values %"),
                    ]
                ),
                dcc.Loading(
                    dcc.Graph(id="target-histogram", config={"displayModeBar": True, "displaylogo": False}),
                    type="circle",
                    color="#4361ee",
                ),
            ],
            className="section-card",
        ),
    ],
    className="mb-4",
)

section_correlation = html.Div(
    id="correlation",
    children=[
        dbc.Card(
            [
                html.H5("🔗 Correlation Matrix"),
                html.Hr(className="section-divider"),
                dcc.Loading(
                    dcc.Graph(id="corr-heatmap", config={"displayModeBar": True, "displaylogo": False}),
                    type="circle",
                    color="#4361ee",
                ),
                dbc.Button(
                    "Show Top 5 Correlated Pairs",
                    id="toggle-corr-table",
                    color="secondary",
                    outline=True,
                    size="sm",
                    className="mt-2 mb-2",
                ),
                dbc.Collapse(
                    html.Div(id="corr-table-container"),
                    id="corr-collapse",
                    is_open=False,
                ),
            ],
            className="section-card",
        ),
    ],
    className="mb-4",
)

section_variance = html.Div(
    id="variance",
    children=[
        dbc.Card(
            [
                html.H5("📉 Variance Thresholding"),
                html.Hr(className="section-divider"),
                html.Label("Variance Threshold:", className="mb-1", style={"fontWeight": "500", "fontSize": "0.85rem"}),
                dcc.Slider(
                    id="variance-slider",
                    min=0,
                    max=1,
                    step=0.01,
                    value=0.1,
                    marks={i / 10: str(round(i / 10, 1)) for i in range(0, 11)},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
                dcc.Loading(
                    dcc.Graph(id="variance-chart", config={"displayModeBar": True, "displaylogo": False}),
                    type="circle",
                    color="#4361ee",
                ),
                html.Div(id="variance-summary", className="mt-2", style={"fontSize": "0.88rem", "color": "#555"}),
            ],
            className="section-card",
        ),
    ],
    className="mb-4",
)

section_mutual_info = html.Div(
    id="mutual-info",
    children=[
        dbc.Card(
            [
                html.H5("🎯 Mutual Information"),
                html.Hr(className="section-divider"),
                html.Label("Target Column:", className="mb-1", style={"fontWeight": "500", "fontSize": "0.85rem"}),
                dcc.Dropdown(id="mi-target-dropdown", clearable=False, style={"maxWidth": "300px"}),
                dcc.Loading(
                    dcc.Graph(id="mi-chart", config={"displayModeBar": True, "displaylogo": False}),
                    type="circle",
                    color="#4361ee",
                ),
            ],
            className="section-card",
        ),
    ],
    className="mb-4",
)

section_pca = html.Div(
    id="pca",
    children=[
        dbc.Card(
            [
                html.H5("🧬 PCA Analysis"),
                html.Hr(className="section-divider"),
                html.Label("Number of Components:", className="mb-1", style={"fontWeight": "500", "fontSize": "0.85rem"}),
                dcc.Slider(
                    id="pca-slider",
                    min=2,
                    max=10,
                    step=1,
                    value=5,
                    marks={i: str(i) for i in range(2, 11)},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Loading(
                                dcc.Graph(id="pca-scree", config={"displayModeBar": True, "displaylogo": False}),
                                type="circle",
                                color="#4361ee",
                            ),
                            md=6,
                        ),
                        dbc.Col(
                            dcc.Loading(
                                dcc.Graph(id="pca-scatter", config={"displayModeBar": True, "displaylogo": False}),
                                type="circle",
                                color="#4361ee",
                            ),
                            md=6,
                        ),
                    ]
                ),
            ],
            className="section-card",
        ),
    ],
    className="mb-4",
)

section_features = html.Div(
    id="features",
    children=[
        dbc.Card(
            [
                html.H5("⚙️ Feature Generation"),
                html.Hr(className="section-divider"),
                dbc.RadioItems(
                    id="feature-method",
                    options=[
                        {"label": "Polynomial (degree 2)", "value": "poly"},
                        {"label": "Interaction Terms Only", "value": "interaction"},
                        {"label": "Log Transform", "value": "log"},
                    ],
                    value="poly",
                    inline=True,
                    className="mb-3",
                    inputClassName="me-1",
                    labelClassName="me-4",
                ),
                dbc.Button("Generate Features", id="btn-generate", color="primary", className="mb-3"),
                dcc.Loading(
                    html.Div(id="feature-table-container"),
                    type="circle",
                    color="#4361ee",
                ),
                html.Div(id="feature-summary", className="mt-2", style={"fontSize": "0.88rem", "color": "#555"}),
            ],
            className="section-card",
        ),
    ],
    className="mb-4",
)

section_shap = html.Div(
    id="shap",
    children=[
        dbc.Card(
            [
                html.H5("🏆 Feature Importance (Permutation)"),
                html.Hr(className="section-divider"),
                dcc.Loading(
                    dcc.Graph(id="importance-chart", config={"displayModeBar": True, "displaylogo": False}),
                    type="circle",
                    color="#4361ee",
                ),
            ],
            className="section-card",
        ),
    ],
    className="mb-4",
)

section_export = html.Div(
    id="export",
    children=[
        dbc.Card(
            [
                html.H5("📥 Export Processed Data"),
                html.Hr(className="section-divider"),
                html.Label("Include transformations:", className="mb-2", style={"fontWeight": "500", "fontSize": "0.85rem"}),
                dbc.Checklist(
                    id="export-checklist",
                    options=[
                        {"label": "  Original features", "value": "original"},
                        {"label": "  Polynomial features", "value": "polynomial"},
                        {"label": "  PCA components", "value": "pca"},
                        {"label": "  Remove low-variance features", "value": "low_variance"},
                    ],
                    value=["original"],
                    className="export-checklist mb-3",
                    inputClassName="me-2",
                ),
                dbc.Button("Download CSV", id="btn-download", color="primary", className="me-2"),
                dcc.Download(id="download-csv"),
                html.Div(id="export-status", className="mt-2", style={"fontSize": "0.85rem", "color": "#0d9488"}),
            ],
            className="section-card",
        ),
    ],
    className="mb-4",
)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════
app.layout = html.Div(
    [
        # Stores — dataset-key-store holds the key string, data loaded from memory
        dcc.Store(id="dataset-key-store", data="ames"),
        dcc.Store(id="variance-threshold-store", data=0.1),

        # Navbar
        navbar,

        # Body
        dbc.Container(
            dbc.Row(
                [
                    # Sidebar
                    dbc.Col(sidebar, md=2, className="p-0"),
                    # Main content
                    dbc.Col(
                        html.Div(
                            [
                                section_overview,
                                section_correlation,
                                section_variance,
                                section_mutual_info,
                                section_pca,
                                section_features,
                                section_shap,
                                section_export,
                            ],
                            className="p-3",
                        ),
                        md=10,
                    ),
                ],
            ),
            fluid=True,
            className="px-0",
        ),
    ]
)


# ═══════════════════════════════════════════════════════════════════════════════
#  CALLBACKS
# ═══════════════════════════════════════════════════════════════════════════════

# ── 1. Dataset selector → Store ────────────────────────────────────────────────
@app.callback(
    Output("dataset-key-store", "data"),
    Input("dataset-selector", "value"),
)
def update_dataset_store(dataset_key):
    return dataset_key


# ── 2a. Sidebar quick stats ───────────────────────────────────────────────────
@app.callback(
    Output("sidebar-stats", "children"),
    Input("dataset-key-store", "data"),
)
def update_sidebar_stats(dataset_key):
    if not dataset_key:
        return no_update
    stats = CACHE[dataset_key]["stats"]
    return html.Div([
        html.Div(f"Rows: {stats['rows']}", style={"fontSize": "0.82rem"}),
        html.Div(f"Columns: {stats['cols']}", style={"fontSize": "0.82rem"}),
        html.Div(f"Numeric: {stats['numeric']}", style={"fontSize": "0.82rem"}),
        html.Div(f"Missing: {stats['missing_pct']}%", style={"fontSize": "0.82rem"}),
    ])


# ── 2b. Overview — metric cards + histogram ───────────────────────────────────
@app.callback(
    Output("stat-rows", "children"),
    Output("stat-cols", "children"),
    Output("stat-numeric", "children"),
    Output("stat-missing", "children"),
    Output("target-histogram", "figure"),
    Input("dataset-key-store", "data"),
)
def update_overview(dataset_key):
    if not dataset_key:
        return "—", "—", "—", "—", _empty_fig()
    stats = CACHE[dataset_key]["stats"]
    fig = CACHE[dataset_key]["histogram"]
    return (
        str(stats["rows"]),
        str(stats["cols"]),
        str(stats["numeric"]),
        f"{stats['missing_pct']}%",
        fig,
    )


# ── 2c. Correlation heatmap + table ──────────────────────────────────────────
@app.callback(
    Output("corr-heatmap", "figure"),
    Output("corr-table-container", "children"),
    Input("dataset-key-store", "data"),
)
def update_correlation(dataset_key):
    if not dataset_key:
        return _empty_fig(), html.Div()
    fig = CACHE[dataset_key]["corr_heatmap"]
    pairs_df = CACHE[dataset_key]["corr_pairs"]
    table = dbc.Table.from_dataframe(pairs_df, striped=True, bordered=False, hover=True, size="sm", className="mt-2")
    return fig, table


@app.callback(
    Output("corr-collapse", "is_open"),
    Input("toggle-corr-table", "n_clicks"),
    State("corr-collapse", "is_open"),
    prevent_initial_call=True,
)
def toggle_corr_collapse(n, is_open):
    return not is_open


# ── 3. Variance slider → chart (fast — no ML involved) ─────────────────────
@app.callback(
    Output("variance-chart", "figure"),
    Output("variance-summary", "children"),
    Input("dataset-key-store", "data"),
    Input("variance-slider", "value"),
)
def update_variance(dataset_key, threshold):
    if not dataset_key:
        return _empty_fig(), ""
    df = _get_df(dataset_key).copy()
    target = TARGET_MAP.get(dataset_key)
    if target and target in df.columns:
        df = df.drop(columns=[target])
    fig, kept, removed = variance_bar_chart(df, threshold)
    summary = f"✅ {kept} features kept, ❌ {removed} features removed (threshold = {threshold})"
    return fig, summary


# ── 4. Mutual info dropdown + chart ─────────────────────────────────────────
@app.callback(
    Output("mi-target-dropdown", "options"),
    Output("mi-target-dropdown", "value"),
    Input("dataset-key-store", "data"),
)
def update_mi_dropdown(dataset_key):
    if not dataset_key:
        return [], None
    df = _get_df(dataset_key)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    options = [{"label": c, "value": c} for c in numeric_cols]
    default = TARGET_MAP.get(dataset_key, numeric_cols[-1] if numeric_cols else None)
    return options, default


@app.callback(
    Output("mi-chart", "figure"),
    Input("dataset-key-store", "data"),
    Input("mi-target-dropdown", "value"),
)
def update_mi_chart(dataset_key, target_col):
    if not dataset_key or not target_col:
        return _empty_fig()
    # Use cache if target matches the default target
    default_target = TARGET_MAP.get(dataset_key)
    if target_col == default_target:
        return CACHE[dataset_key]["mi_chart"]
    # Otherwise compute live (only when user picks a different target)
    df = _get_df(dataset_key)
    return mutual_info_chart(df, target_col)


# ── 5. PCA slider → scree + scatter ─────────────────────────────────────────
@app.callback(
    Output("pca-scree", "figure"),
    Output("pca-scatter", "figure"),
    Input("dataset-key-store", "data"),
    Input("pca-slider", "value"),
)
def update_pca(dataset_key, n_components):
    if not dataset_key:
        return _empty_fig(), _empty_fig()
    # Use cache for default (5 components)
    if n_components == 5:
        return CACHE[dataset_key]["pca_5"]
    df = _get_df(dataset_key)
    target = TARGET_MAP.get(dataset_key, df.columns[-1])
    return pca_figures(df, target, n_components)


# ── 6. Feature generation (user-triggered, fast) ───────────────────────────
@app.callback(
    Output("feature-table-container", "children"),
    Output("feature-summary", "children"),
    Input("btn-generate", "n_clicks"),
    State("dataset-key-store", "data"),
    State("feature-method", "value"),
    prevent_initial_call=True,
)
def update_feature_gen(n_clicks, dataset_key, method):
    if not dataset_key:
        return html.Div("No dataset loaded."), ""
    df = _get_df(dataset_key)
    original_count = len(df.select_dtypes(include="number").columns)
    result = generate_features(df, method)
    new_count = len(result.columns) - original_count
    total = len(result.columns)

    preview = result.head(5).round(4)
    table = dbc.Table.from_dataframe(
        preview,
        striped=True,
        bordered=False,
        hover=True,
        size="sm",
        responsive=True,
        style={"fontSize": "0.8rem"},
    )
    summary = f"✨ Generated {new_count} new features. Total features: {total}"
    return table, summary


# ── 7. Feature importance (cached — instant) ───────────────────────────────
@app.callback(
    Output("importance-chart", "figure"),
    Input("dataset-key-store", "data"),
)
def update_importance(dataset_key):
    if not dataset_key:
        return _empty_fig()
    return CACHE[dataset_key]["importance"]


# ── 8. Export CSV ──────────────────────────────────────────────────────────
@app.callback(
    Output("download-csv", "data"),
    Output("export-status", "children"),
    Input("btn-download", "n_clicks"),
    State("dataset-key-store", "data"),
    State("export-checklist", "value"),
    State("variance-slider", "value"),
    prevent_initial_call=True,
)
def download_csv(n_clicks, dataset_key, selections, var_threshold):
    if not dataset_key or not selections:
        return no_update, "⚠️ Please select at least one transformation."
    df = _get_df(dataset_key)
    csv_string = build_export_df(df, selections, var_threshold)
    return (
        dict(content=csv_string, filename="autofe_processed_features.csv"),
        "✅ CSV downloaded successfully!",
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  RUN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app.run(debug=True, port=8050)
