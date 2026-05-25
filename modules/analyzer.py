"""
analyzer.py
-----------------------
Computes descriptive statistics, missing value analysis,
correlations, and anomaly detection on a LoadedDataset.

Usage:
    from modules.analyzer import analyze
    from modules.loader   import load_file

    dataset = load_file(uploaded_file)
    report  = analyze(dataset)

    report.summary_df       # descriptive statistics table
    report.missing_df       # missing values table
    report.correlation_df   # Pearson correlation matrix
    report.anomalies_df     # rows flagged as anomalies
    report.insights         # list of plain-English insight strings
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from modules.loader import LoadedDataset

#  Output data structure
@dataclass
class AnalysisReport:
    summary_df: pd.DataFrame                  # descriptive stats per numeric column
    missing_df: pd.DataFrame                  # missing values per column
    correlation_df: Optional[pd.DataFrame]        # Pearson matrix (None if <2 numeric cols)
    anomalies_df: pd.DataFrame                  # rows that contain at least one anomaly
    insights: list[str] = field(default_factory=list)  # auto-generated plain text

#  Main entry point
def analyze(dataset: LoadedDataset) -> AnalysisReport:

    # Runs the full analysis pipeline on a LoadedDataset.
    # Returns an AnalysisReport with all results ready for display.

    if not dataset.ok or dataset.df is None:
        # Return an empty report if the dataset is invalid
        return AnalysisReport(
            summary_df=pd.DataFrame(),
            missing_df=pd.DataFrame(),
            correlation_df=None,
            anomalies_df=pd.DataFrame(),
            insights=["No valid dataset available for analysis."],
        )

    df = dataset.df
    summary_df = _compute_summary(df, dataset.numeric_cols)
    missing_df = _compute_missing(df)
    correlation_df = _compute_correlation(df, dataset.numeric_cols)
    anomalies_df = _detect_anomalies(df, dataset.numeric_cols)
    insights = _generate_insights(df, dataset, summary_df, missing_df, anomalies_df)

    return AnalysisReport(
        summary_df=summary_df,
        missing_df=missing_df,
        correlation_df=correlation_df,
        anomalies_df=anomalies_df,
        insights=insights,
    )

#  1. Descriptive statistics
def _compute_summary(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:

    # Computes descriptive statistics for every numeric column:
    # count, mean, median, std, min, max, skewness, and kurtosis.
    # Returns a formatted DataFrame with one row per column.

    if not numeric_cols:
        return pd.DataFrame()

    rows = []
    for col in numeric_cols:
        series = df[col].dropna()

        rows.append({
            "Column": col,
            "Count": int(series.count()),
            "Mean": round(series.mean(), 4),
            "Median": round(series.median(), 4),
            "Std dev": round(series.std(), 4),
            "Min": round(series.min(), 4),
            "Max": round(series.max(), 4),
            "Skewness": round(float(stats.skew(series)), 4),
            "Kurtosis": round(float(stats.kurtosis(series)), 4),
        })

    return pd.DataFrame(rows).set_index("Column")

#  2. Missing value analysis
def _compute_missing(df: pd.DataFrame) -> pd.DataFrame:

    # Counts the number of missing values per column and reports them as a percentage,
    # from worst to best. Columns with no missing values are excluded.

    total = len(df)
    rows  = []

    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        if n_missing == 0:
            continue
        rows.append({
            "Column": col,
            "Missing": n_missing,
            "Missing (%)": round(n_missing / total * 100, 2),
            "Dtype": str(df[col].dtype),
        })

    if not rows:
        return pd.DataFrame(columns=["Column", "Missing", "Missing (%)", "Dtype"])

    return (
        pd.DataFrame(rows)
        .set_index("Column")
        .sort_values("Missing (%)", ascending=False)
    )

#  3. Correlation matrix
def _compute_correlation(df: pd.DataFrame, numeric_cols: list[str]) -> Optional[pd.DataFrame]:

    # Calculates the correlation for all numeric columns.
    # Returns `None` if there are fewer than 2 numeric columns (in which case the matrix is meaningless).
    # Values are rounded to 2 decimal places for readability.

    if len(numeric_cols) < 2:
        return None

    corr = df[numeric_cols].corr(method="pearson").round(2)
    return corr

#  4. Anomaly detection (IQR method)
def _detect_anomalies(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:

    # Flags rows that contain at least one statistical outlier using the
    # IQR (Interquartile Range) method:
    #    lower fence = Q1 - 1.5 * IQR
    #    upper fence = Q3 + 1.5 * IQR
    # Any value outside the fences is considered an anomaly.
    # Returns a copy of the flagged rows plus an extra column
    # 'anomaly_in' that lists which columns triggered the flag.

    if not numeric_cols:
        return pd.DataFrame()

    outlier_mask = pd.Series(False, index=df.index)
    anomaly_cols_per_row: dict[int, list[str]] = {}

    for col in numeric_cols:
        series = df[col].dropna()

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        col_mask = (df[col] < lower) | (df[col] > upper)

        # Record which column triggered each outlier row
        for idx in df.index[col_mask]:
            anomaly_cols_per_row.setdefault(idx, []).append(col)

        outlier_mask = outlier_mask | col_mask

    if not outlier_mask.any():
        return pd.DataFrame()

    result = df[outlier_mask].copy()
    result["anomaly_in"] = result.index.map(
        lambda i: ", ".join(anomaly_cols_per_row.get(i, []))
    )
    return result

#  5. Auto-generated insights
def _generate_insights(
    df: pd.DataFrame,
    dataset: LoadedDataset,
    summary_df: pd.DataFrame,
    missing_df: pd.DataFrame,
    anomalies_df: pd.DataFrame,
) -> list[str]:

    # Produces a list of plain-English sentences that summarise the
    # most important findings. These are consumed by agent.py to give
    # the AI model context without passing the entire DataFrame.

    insights: list[str] = []
    n_rows, n_cols = df.shape

    # -- Dataset size --
    insights.append(f"The dataset has {n_rows:,} rows and {n_cols} columns.")

    # -- Missing values --
    if missing_df.empty:
        insights.append("No missing values detected.")
    else:
        worst_col = missing_df["Missing (%)"].idxmax()
        worst_pct = missing_df["Missing (%)"].max()
        insights.append(
            f"{len(missing_df)} column(s) have missing values. "
            f"Worst: '{worst_col}' ({worst_pct}% missing)."
        )

    # -- Highly skewed columns --
    if not summary_df.empty and "Skewness" in summary_df.columns:
        skewed = summary_df[summary_df["Skewness"].abs() > 1]
        if not skewed.empty:
            names = ", ".join(f"'{c}'" for c in skewed.index)
            insights.append(
                f"Highly skewed columns (|skewness| > 1): {names}. "
                f"Consider log-transforming before modelling."
            )

    # -- Strong correlations --
    if dataset.meta.get("semantic_summary"):
        numeric_cols = dataset.numeric_cols
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr().abs()
            # Zero out the diagonal via arithmetic (avoids numpy in-place write on CoW arrays)
            corr_no_diag = corr - pd.DataFrame(
                np.eye(len(corr)), index=corr.index, columns=corr.columns
            )
            # Find the pair with the highest correlation
            max_corr = corr_no_diag.max().max()
            if max_corr > 0.7:
                col_a = corr_no_diag.max(axis=1).idxmax()
                col_b = corr_no_diag[col_a].idxmax()
                insights.append(
                    f"Strong correlation ({max_corr:.2f}) detected between "
                    f"'{col_a}' and '{col_b}'."
                )

    # -- Anomalies --
    if anomalies_df.empty:
        insights.append("No statistical anomalies detected (IQR method).")
    else:
        pct = round(len(anomalies_df) / len(df) * 100, 1)
        insights.append(
            f"{len(anomalies_df)} row(s) flagged as anomalies ({pct}% of data)."
        )

    # -- Constant columns (zero variance) --
    if not summary_df.empty and "Std dev" in summary_df.columns:
        constant = summary_df[summary_df["Std dev"] == 0]
        if not constant.empty:
            names = ", ".join(f"'{c}'" for c in constant.index)
            insights.append(
                f"Column(s) with zero variance (constant value): {names}. "
                f"These carry no information and can be dropped."
            )

    return insights