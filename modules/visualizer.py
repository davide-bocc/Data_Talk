"""
visualizer.py
------------------------
Generates interactive Plotly charts from a LoadedDataset
and an AnalysisReport. Every function returns a plotly Figure
that can be displayed in Streamlit with st.plotly_chart().

Usage:
    from modules.visualizer import (
        plot_distribution,
        plot_correlation_heatmap,
        plot_missing_values,
        plot_categorical_bar,
        plot_time_series,
        plot_scatter,
        auto_charts,
    )

    fig = plot_distribution(dataset.df, "revenue")
    st.plotly_chart(fig, use_container_width=True)
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modules.loader import LoadedDataset
from modules.analyzer import AnalysisReport

#  -- Global style constants --

# Consistent color palette used across all charts
PALETTE = px.colors.qualitative.Safe
PRIMARY_COLOR = "#4C72B0"
DANGER_COLOR = "#DD4444"

# Default layout applied to every figure
_BASE_LAYOUT = dict(
    paper_bgcolor = "rgba(0,0,0,0)",   # transparent background
    plot_bgcolor = "rgba(0,0,0,0)",
    font = dict(family="Inter, sans-serif", size=13),
    margin = dict(l=40, r=40, t=60, b=40),
    hoverlabel = dict(bgcolor="white", font_size=13),
)

def _apply_style(fig: go.Figure, title: str = "") -> go.Figure:
    # Applies the base layout and an optional title to any figure.
    fig.update_layout(title=dict(text=title, font=dict(size=16)), **_BASE_LAYOUT)
    return fig

#  1. Distribution - histogram + box plot
def plot_distribution(df: pd.DataFrame, column: str) -> go.Figure:

    # Combined histogram and box plot for a single numeric column.
    # Uses a two-row subplot: histogram on top, box plot below.
    # The box plot shares the same x-axis so they are perfectly aligned.

    series = df[column].dropna()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.80, 0.20],
        vertical_spacing=0.02,
    )

    # -- Top row: histogram --
    fig.add_trace(
        go.Histogram(
            x=series,
            nbinsx=40,
            marker_color=PRIMARY_COLOR,
            opacity=0.85,
            name="Distribution",
            hovertemplate="Range: %{x}<br>Count: %{y}<extra></extra>",
        ),
        row=1, col=1,
    )

    # -- Bottom row: box plot --
    fig.add_trace(
        go.Box(
            x=series,
            marker_color=PRIMARY_COLOR,
            boxmean="sd",        # also draw the mean and ±1 std deviation
            name="Box plot",
            hovertemplate="Value: %{x}<extra></extra>",
        ),
        row=2, col=1,
    )

    # -- Vertical lines for mean and median --
    mean = series.mean()
    median = series.median()

    for value, label, color in [
        (mean,   f"Mean: {mean:.2f}",     "orange"),
        (median, f"Median: {median:.2f}", "green"),
    ]:
        fig.add_vline(
            x=value,
            line_dash="dash",
            line_color=color,
            annotation_text=label,
            annotation_position="top right",
            row=1, col=1,
        )

    fig.update_yaxes(title_text="Count", row=1, col=1)
    return _apply_style(fig, title=f"Distribution — {column}")

#  2. Correlation heatmap
def plot_correlation_heatmap(correlation_df: pd.DataFrame) -> go.Figure:

    # Renders the Pearson correlation matrix as an annotated heatmap.
    # Values range from -1 (dark red) to +1 (dark blue).
    # The upper triangle is hidden to avoid redundancy.

    # -- Mask the upper triangle - it mirrors the lower triangle exactly --
    import numpy as np
    mask = np.triu(np.ones(correlation_df.shape, dtype=bool), k=1)
    masked = correlation_df.copy().astype(float)
    masked[mask] = None     # None values are not rendered by Plotly

    fig = go.Figure(
        go.Heatmap(
            z=masked.values,
            x=masked.columns.tolist(),
            y=masked.index.tolist(),
            colorscale="RdBu",          # red = negative, white = 0, blue = positive
            zmid=0,                     # centre the colour scale at 0
            zmin=-1,
            zmax=1,
            text=masked.round(2).values,
            texttemplate="%{text}",     # show numeric value in each cell
            hovertemplate="x: %{x}<br>y: %{y}<br>correlation: %{z:.2f}<extra></extra>",
            showscale=True,
            colorbar=dict(title="Pearson r"),
        )
    )

    # -- Square cells regardless of how many columns there are --
    fig.update_layout(
        width=max(400, 80 * len(correlation_df.columns)),
        height=max(400, 80 * len(correlation_df.columns)),
    )

    return _apply_style(fig, title="Correlation Matrix (Pearson)")

#  3. Missing values bar chart
def plot_missing_values(missing_df: pd.DataFrame) -> Optional[go.Figure]:

    # Horizontal bar chart showing missing-value percentage per column.
    # Returns None if there are no missing values (nothing to plot).
    # Bars are coloured green → orange → red based on severity.

    if missing_df.empty:
        return None

    df_plot = missing_df.reset_index().sort_values("Missing (%)", ascending=True)

    # -- Assign a colour based on severity thresholds --
    def _color(pct: float) -> str:
        if pct < 5:
            return "#2ecc71"    # green  — acceptable
        if pct < 30:
            return "#f39c12"    # orange — moderate
        return DANGER_COLOR     # red    — critical

    colors = [_color(p) for p in df_plot["Missing (%)"]]

    fig = go.Figure(
        go.Bar(
            x=df_plot["Missing (%)"],
            y=df_plot["Column"],
            orientation="h",
            marker_color=colors,
            hovertemplate="<b>%{y}</b><br>Missing: %{x}%<extra></extra>",
        )
    )

    # -- Reference line at 30% — common threshold for dropping a column --
    fig.add_vline(
        x=30,
        line_dash="dot",
        line_color="grey",
        annotation_text="30% threshold",
        annotation_position="top right",
    )

    fig.update_xaxes(title_text="Missing (%)", range=[0, 100])
    return _apply_style(fig, title="Missing Values by Column")

#  4. Categorical bar chart
def plot_categorical_bar(df: pd.DataFrame, column: str, top_n: int = 15) -> go.Figure:

    # Counts occurrences of each category in a column and plots a
    # vertical bar chart, limited to the top_n most frequent values.
    # Categories are sorted from most to least frequent so the user
    # immediately sees what dominates.

    counts = (
        df[column]
        .value_counts()
        .head(top_n)
        .reset_index()
    )
    counts.columns = ["Category", "Count"]

    fig = px.bar(
        counts,
        x="Category",
        y="Count",
        color="Count",
        color_continuous_scale=["#aec6e8", PRIMARY_COLOR],
        text="Count",
    )

    fig.update_traces(
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>",
    )
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(tickangle=-30)

    return _apply_style(fig, title=f"Category Counts — {column} (top {top_n})")

#  5. Time series line chart
def plot_time_series(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    resample_freq: str = "auto",
) -> go.Figure:
    """
    Plots a numeric column over time as a line chart.
    Optionally resamples the data to reduce noise.

    resample_freq:
        "auto"  - chooses daily / weekly / monthly based on date range
        "D"     - daily
        "W"     - weekly
        "ME"    - month-end
        "QE"    - quarter-end
        None    - no resampling, plot raw values
    """
    ts = df[[date_col, value_col]].dropna().copy()
    ts[date_col] = pd.to_datetime(ts[date_col])
    ts = ts.sort_values(date_col)

    # -- Auto-select frequency based on total date range --
    if resample_freq == "auto":
        span_days = (ts[date_col].max() - ts[date_col].min()).days
        if span_days <= 60:
            resample_freq = None        # raw daily data — no need to resample
        elif span_days <= 365:
            resample_freq = "W"
        elif span_days <= 365 * 3:
            resample_freq = "ME"
        else:
            resample_freq = "QE"

    if resample_freq:
        ts = (
            ts.set_index(date_col)
            .resample(resample_freq)[value_col]
            .mean()
            .reset_index()
        )

    fig = go.Figure(
        go.Scatter(
            x=ts[date_col],
            y=ts[value_col],
            mode="lines+markers",
            line=dict(color=PRIMARY_COLOR, width=2),
            marker=dict(size=4),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>",
            name=value_col,
        )
    )

    freq_label = {"W": "weekly", "ME": "monthly", "QE": "quarterly"}.get(
        resample_freq, "raw"
    )
    fig.update_yaxes(title_text=value_col)
    return _apply_style(
        fig,
        title=f"Time Series — {value_col} over {date_col} ({freq_label})",
    )

#  6. Scatter plot with optional colour grouping
def plot_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None,
    trendline: bool = True,
) -> go.Figure:

    # Scatter plot between two numeric columns.
    # Optionally colours points by a categorical column.
    # Optionally draws an OLS trendline per group.

    _trendline = None
    if trendline:
        try:
            import statsmodels.api  # noqa: F401
            _trendline = "ols"
        except ImportError:
            pass

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        trendline=_trendline,
        color_discrete_sequence=PALETTE,
        opacity=0.7,
        hover_data=df.columns.tolist(),
    )

    fig.update_traces(marker=dict(size=6))
    return _apply_style(fig, title=f"Scatter — {x_col} vs {y_col}")

#  7. Auto-charts — main entry point
def auto_charts(
    dataset: LoadedDataset,
    report:  AnalysisReport,
) -> dict[str, go.Figure]:
    """
    Inspects the dataset and automatically generates the most relevant
    charts without any user input. Returns an ordered dictionary where
    keys are human-readable chart titles and values are Plotly figures.

    Chart selection logic:
        - Always: missing-value bar (if any missing)
        - Always: correlation heatmap (if ≥2 numeric cols)
        - Per numeric column (up to 4): distribution histogram+box
        - Per categorical column (up to 3): category bar chart
        - Per datetime × numeric pair (up to 2): time series line chart
    """
    charts: dict[str, go.Figure] = {}
    df = dataset.df

    # -- Missing values --
    if not report.missing_df.empty:
        charts["Missing Values"] = plot_missing_values(report.missing_df)

    # -- Correlation heatmap --
    if report.correlation_df is not None:
        charts["Correlation Matrix"] = plot_correlation_heatmap(report.correlation_df)

    # -- Distributions (first 4 numeric columns) --
    for col in dataset.numeric_cols[:4]:
        charts[f"Distribution — {col}"] = plot_distribution(df, col)

    # -- Categorical bars (first 3 categorical columns) --
    for col in dataset.categorical_cols[:3]:
        charts[f"Categories — {col}"] = plot_categorical_bar(df, col)

    # -- Time series (first datetime × first 2 numeric columns) --
    if dataset.datetime_cols:
        date_col = dataset.datetime_cols[0]
        for val_col in dataset.numeric_cols[:2]:
            charts[f"Time Series — {val_col}"] = plot_time_series(df, date_col, val_col)

    return charts