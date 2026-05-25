"""
forecaster.py
------------------------
Time-series forecasting using Facebook Prophet (primary) with a
scikit-learn Linear Regression fallback when Prophet is unavailable.

Usage:
    from modules.forecaster import forecast
    from modules.loader     import load_file

    dataset = load_file(uploaded_file)
    result  = forecast(dataset, date_col="date", value_col="revenue", periods=30)

    if result.ok:
        st.plotly_chart(result.fig)
        st.dataframe(result.forecast_df)
    else:
        st.error(result.error)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy  as np
import pandas as pd
import plotly.graph_objects as go

from modules.loader import LoadedDataset

#  1. Output data structure
@dataclass
class ForecastResult:
    ok: bool
    forecast_df: Optional[pd.DataFrame] = None   # future predictions table
    fig: Optional[go.Figure] = None   # interactive Plotly chart
    model_used: Optional[str] = None   # "prophet" or "linear"
    error: Optional[str] = None

#  Style constants (mirrors visualizer.py)
_PRIMARY = "#4C72B0"
_FORECAST = "#E87040"
_CONFIDENCE = "rgba(232, 112, 64, 0.15)"

_BASE_LAYOUT = dict(
    paper_bgcolor = "rgba(0,0,0,0)",
    plot_bgcolor = "rgba(0,0,0,0)",
    font = dict(family="Inter, sans-serif", size=13),
    margin = dict(l=40, r=40, t=60, b=40),
    hoverlabel = dict(bgcolor="white", font_size=13),
)

#  2. Main entry point
def forecast(
    dataset:   LoadedDataset,
    date_col:  str,
    value_col: str,
    periods:   int = 30,
    freq:      str = "auto",
) -> ForecastResult:
    """
    Runs a time-series forecast on one numeric column against one
    datetime column. Tries Prophet first; falls back to Linear
    Regression if Prophet is not installed.

    Parameters
    ----------
    dataset: LoadedDataset produced by loader.py
    date_col: name of the datetime column to use as time axis
    value_col: name of the numeric column to forecast
    periods: how many future steps to predict
    freq: pandas offset alias ("D", "W", "ME", "QE") or "auto"
    """
    if not dataset.ok or dataset.df is None:
        return ForecastResult(ok=False, error="Invalid dataset.")

    df = dataset.df[[date_col, value_col]].dropna().copy()

    if len(df) < 10:
        return ForecastResult(
            ok=False,
            error=f"Not enough data to forecast '{value_col}'. "
                  f"At least 10 non-null rows are required (found {len(df)})."
        )

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    if freq == "auto":
        freq = _infer_frequency(df[date_col])

    # -- Try Prophet first, then fall back to Linear Regression --
    try:
        from prophet import Prophet   # noqa: PLC0415 (lazy import intentional)
        return _forecast_prophet(df, date_col, value_col, periods, freq)
    except ImportError:
        try:
            return _forecast_linear(df, date_col, value_col, periods, freq)
        except ImportError:
            return ForecastResult(
                ok=False,
                error="Forecasting requires 'prophet' or 'scikit-learn'. "
                      "Install one: pip install scikit-learn",
            )
    except Exception as exc:
        # Prophet is installed but raised an error — try linear as safety net
        try:
            return _forecast_linear(df, date_col, value_col, periods, freq)
        except Exception:
            return ForecastResult(ok=False, error=f"Forecasting failed: {exc}")

#  3. Frequency inference
def _infer_frequency(dates: pd.Series) -> str:

    # Estimates the most likely sampling frequency of a datetime series
    # by looking at the median gap between consecutive dates.

    deltas = dates.diff().dropna()
    median_days = deltas.median().days

    if median_days <= 1:
        return "D"  # daily
    if median_days <= 8:
        return "W"   # weekly
    if median_days <= 32:
        return "ME"  # month-end
    return "QE"   # quarter-end

#  4. Prophet forecaster
def _forecast_prophet(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    periods: int,
    freq: str,
) -> ForecastResult:
    """
    Fits a Facebook Prophet model and predicts `periods` future steps.

    Prophet requires columns named exactly 'ds' (date) and 'y' (value).
    We rename on entry and rename back on exit so the rest of the code
    never needs to know about Prophet's naming convention.
    """
    from prophet import Prophet

    prophet_df = df.rename(columns={date_col: "ds", value_col: "y"})

    model = Prophet(
        yearly_seasonality = True,
        weekly_seasonality = True,
        daily_seasonality = False,
        interval_width = 0.95,    # 95% confidence interval
    )
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=periods, freq=freq)
    raw = model.predict(future)

    # -- Extract and rename the columns we care about --
    forecast_df = raw[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(
        columns={
            "ds": "date",
            "yhat": "forecast",
            "yhat_lower": "lower_bound",
            "yhat_upper": "upper_bound",
        }
    )

    fig = _build_forecast_chart(
        historical = df.rename(columns={date_col: "date", value_col: "value"}),
        forecast   = forecast_df,
        value_col  = value_col,
        model_name = "Prophet",
    )

    return ForecastResult(
        ok=True,
        forecast_df=forecast_df,
        fig=fig,
        model_used="prophet",
    )

#  5. Linear Regression fallback
def _forecast_linear(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    periods: int,
    freq: str,
) -> ForecastResult:
    """
    Simple OLS linear regression on a numeric time index.
    Used when Prophet is not installed.

    The model fits  y = a * t + b  where t is the number of days
    since the first observation. It then predicts t values for the
    future periods and converts them back to real dates.

    Also computes a ±1.96σ confidence band from the residuals,
    which approximates a 95% prediction interval.
    """
    from sklearn.linear_model import LinearRegression

    dates = pd.to_datetime(df[date_col])
    origin = dates.min()

    # Convert dates to a plain numeric feature: days since origin
    X = (dates - origin).dt.days.values.reshape(-1, 1)
    y = df[value_col].values

    model = LinearRegression()
    model.fit(X, y)

    # Build future date index
    last_date = dates.max()
    future_dates = pd.date_range(start=last_date, periods=periods + 1, freq=freq)[1:]
    X_future = (future_dates - origin).days.values.reshape(-1, 1)
    y_pred = model.predict(X_future)
    residuals = y - model.predict(X)
    sigma = residuals.std()

    forecast_df = pd.DataFrame({
        "date": future_dates,
        "forecast": y_pred.round(4),
        "lower_bound": (y_pred - 1.96 * sigma).round(4),
        "upper_bound": (y_pred + 1.96 * sigma).round(4),
    })

    historical = pd.DataFrame({"date": dates, "value": y})

    fig = _build_forecast_chart(
        historical = historical,
        forecast   = forecast_df,
        value_col  = value_col,
        model_name = "Linear Regression",
    )

    return ForecastResult(
        ok=True,
        forecast_df=forecast_df,
        fig=fig,
        model_used="linear",
    )

#  6. Chart builder (shared by both models)
def _build_forecast_chart(
    historical: pd.DataFrame,
    forecast: pd.DataFrame,
    value_col: str,
    model_name: str,
) -> go.Figure:
    """
    Builds the forecast chart shared by both Prophet and Linear models.
    Composed of three layers:
        1. Shaded confidence band (upper - lower bounds)
        2. Dashed forecast line
        3. Solid historical line
    """
    fig = go.Figure()

    # -- Layer 1: confidence band --
    # We draw it as a filled area by combining upper and lower bounds.
    # The trick: plot upper bound going forward, then lower bound going
    # backward (reversed), so the filled shape closes correctly.
    fig.add_trace(go.Scatter(
        x = pd.concat([forecast["date"], forecast["date"][::-1]]),
        y = pd.concat([forecast["upper_bound"], forecast["lower_bound"][::-1]]),
        fill = "toself",
        fillcolor = _CONFIDENCE,
        line = dict(color="rgba(0,0,0,0)"),   # invisible border
        name = "95% confidence interval",
        hoverinfo = "skip",
    ))

    # -- Layer 2: forecast line --
    fig.add_trace(go.Scatter(
        x = forecast["date"],
        y = forecast["forecast"],
        mode = "lines",
        line = dict(color=_FORECAST, width=2, dash="dash"),
        name = f"Forecast ({model_name})",
        hovertemplate = "%{x|%Y-%m-%d}<br>Forecast: %{y:.2f}<extra></extra>",
    ))

    # -- Layer 3: historical line --
    fig.add_trace(go.Scatter(
        x = historical["date"],
        y = historical["value"],
        mode = "lines+markers",
        line = dict(color=_PRIMARY, width=2),
        marker = dict(size=3),
        name = f"Historical ({value_col})",
        hovertemplate = "%{x|%Y-%m-%d}<br>Value: %{y:.2f}<extra></extra>",
    ))

    # -- Vertical separator between historical and forecast --
    # Plotly 6.x cannot compute annotation position on datetime axes,
    # so the annotation is added separately.
    split_date = historical["date"].max().isoformat()
    fig.add_vline(x=split_date, line_dash="dot", line_color="grey")
    fig.add_annotation(
        x=split_date, y=1, yref="paper",
        text="Forecast start",
        showarrow=False, xanchor="left",
        font=dict(size=11, color="grey"),
    )

    fig.update_layout(
        title = dict(text=f"Forecast — {value_col} ({model_name})", font=dict(size=16)),
        legend = dict(orientation="h", y=-0.15),
        **_BASE_LAYOUT,
    )
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text=value_col)

    return fig

#  7. Utility: list forecastable column pairs
def get_forecastable_pairs(dataset: LoadedDataset) -> list[tuple[str, str]]:

    # Used by app.py to populate the dropdown menus in the forecast section.

    pairs = []
    for date_col in dataset.datetime_cols:
        for val_col in dataset.numeric_cols:
            pairs.append((date_col, val_col))
    return pairs