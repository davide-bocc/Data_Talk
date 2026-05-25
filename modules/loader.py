"""
loader.py
--------------------
Loads and validates CSV / Excel files.

I handled all the encoding logic here because the files that
are frequently uploaded often have issues with Italian characters (à, è, ù).
It automatically detects the data type of each column
(date, number, text) to pass the information on to the other modules.

Usage:
    from modules.loader import load_file, LoadedDataset

    result = load_file(uploaded_file)
    if result.ok:
        df   = result.df
        meta = result.meta
    else:
        st.error(result.error)
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import Optional

import chardet
import pandas as pd


#  Semantic column types
SEMANTIC_TYPES = {
    "numeric_continuous": "Continuous numeric",
    "numeric_discrete": "Discrete numeric",
    "categorical": "Categorical",
    "datetime": "Date / time",
    "text_free": "Free text",
    "boolean": "Boolean",
    "id_like": "Identifier (ID)",
    "unknown": "Unknown",
}

# Regex: column names that hint at a datetime value
_DT_NAME_HINTS = re.compile(
    r"(date|time|timestamp|day|month|year|week|hour|period|data|giorno|mese|anno|ora)",
    re.IGNORECASE,
)

# Regex: column names that hint at an identifier
_ID_NAME_HINTS = re.compile(
    r"(^id$|_id$|^cod|^code|^key$|^uuid|^serial|^number$|^num$|^numero)",
    re.IGNORECASE,
)

#  Output data structures
@dataclass
class ColumnMeta:
    name: str
    dtype_raw: str          # original pandas dtype
    semantic: str           # key from SEMANTIC_TYPES
    n_unique: int
    n_missing: int
    pct_missing: float
    sample_values: list     # up to 5 non-null example values


@dataclass
class LoadedDataset:
    ok: bool
    df: Optional[pd.DataFrame] = None
    meta: dict = field(default_factory=dict)
    columns: list[ColumnMeta] = field(default_factory=list)
    error: Optional[str] = None

    # Shortcuts for other modules
    @property
    def numeric_cols(self) -> list[str]:
        return [c.name for c in self.columns
                if c.semantic in ("numeric_continuous", "numeric_discrete")]

    @property
    def categorical_cols(self) -> list[str]:
        return [c.name for c in self.columns if c.semantic == "categorical"]

    @property
    def datetime_cols(self) -> list[str]:
        return [c.name for c in self.columns if c.semantic == "datetime"]

    @property
    def text_cols(self) -> list[str]:
        return [c.name for c in self.columns if c.semantic == "text_free"]

#  Main entry point
def load_file(uploaded_file) -> LoadedDataset:

    # Receives a file from st.file_uploader and returns a LoadedDataset
    # containing the DataFrame and all metadata ready for use.

    try:
        raw_bytes = uploaded_file.read()
        filename = uploaded_file.name.lower()

        if filename.endswith(".csv"):
            df = _read_csv(raw_bytes)
        elif filename.endswith((".xlsx", ".xls")):
            df = _read_excel(raw_bytes, filename)
        else:
            return LoadedDataset(
                ok=False,
                error=f"Unsupported format: '{uploaded_file.name}'. "
                      f"Please upload a .csv, .xlsx or .xls file."
            )

        df = _clean_dataframe(df)
        columns = _analyze_columns(df)
        meta = _build_meta(df, columns, uploaded_file.name)

        return LoadedDataset(ok=True, df=df, meta=meta, columns=columns)

    except Exception as exc:
        return LoadedDataset(ok=False, error=f"Error while loading the file: {exc}")

#  File readers
def _read_csv(raw: bytes) -> pd.DataFrame:

    # Tries multiple encodings and separators until the CSV is read correctly.
    # Handles Italian files (semicolon separator, latin-1 encoding).

    detected = chardet.detect(raw)
    encodings = list({detected.get("encoding", "utf-8"), "utf-8", "latin-1", "cp1252"})
    separators = [",", ";", "\t", "|"]

    last_exc = None
    for enc in encodings:
        for sep in separators:
            try:
                df = pd.read_csv(
                    io.BytesIO(raw),
                    sep=sep,
                    encoding=enc,
                    on_bad_lines="warn",
                    low_memory=False,
                )
                if df.shape[1] > 1:     # discard if only one column (wrong separator)
                    return df
            except Exception as e:
                last_exc = e

    # Last resort fallback
    try:
        return pd.read_csv(io.BytesIO(raw), encoding="utf-8", on_bad_lines="warn")
    except Exception as e:
        raise ValueError(f"Could not read CSV file: {e}") from last_exc


def _read_excel(raw: bytes, filename: str) -> pd.DataFrame:
    # For now, I'm only reading the first sheet of an Excel file.
    # You could add a dropdown menu to select the sheet you want to view

    engine = "openpyxl" if filename.endswith(".xlsx") else "xlrd"
    return pd.read_excel(io.BytesIO(raw), engine=engine, sheet_name=0)

#  DataFrame cleaning
def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:

    # Basic cleaning: normalise column headers, drop empty rows/columns,
    # and attempt to convert object columns to datetime or numeric types.

    # Strip whitespace from column names
    df.columns = [str(c).strip() for c in df.columns]

    # Drop rows and columns that are entirely empty
    df.dropna(how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)

    # Try to convert object columns to a better type
    # pandas 3.0 uses StringDtype ("str") instead of "object" for string columns
    for col in df.select_dtypes(include=["object", "str"]).columns:

        # Attempt datetime conversion first
        try:
            converted = pd.to_datetime(df[col], format="mixed", errors="coerce")
            if converted.notna().mean() > 0.7:      # accept if ≥70% values parse cleanly
                df[col] = converted
                continue
        except Exception:
            pass

        # Attempt numeric conversion (handles Italian decimal comma: 1.234,56)
        try:
            cleaned = (
                df[col]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .str.replace(r"[^\d.\-+eE]", "", regex=True)
            )
            converted_num = pd.to_numeric(cleaned, errors="coerce")
            if converted_num.notna().mean() > 0.7:
                df[col] = converted_num
        except Exception:
            pass

    return df

#  Column analysis
def _analyze_columns(df: pd.DataFrame) -> list[ColumnMeta]:

    # Iterates over every column and builds a ColumnMeta object
    # containing statistics and the inferred semantic type.

    results = []
    n_rows  = len(df)

    for col in df.columns:
        series = df[col]
        n_miss = int(series.isna().sum())
        pct_miss = round(n_miss / n_rows * 100, 1) if n_rows else 0.0
        n_unique = int(series.nunique(dropna=True))
        samples = series.dropna().head(5).tolist()
        semantic = _infer_semantic(series, col, n_unique, n_rows)

        results.append(ColumnMeta(
            name=col,
            dtype_raw=str(series.dtype),
            semantic=semantic,
            n_unique=n_unique,
            n_missing=n_miss,
            pct_missing=pct_miss,
            sample_values=samples,
        ))

    return results


def _infer_semantic(
    series: pd.Series,
    col_name: str,
    n_unique: int,
    n_rows: int,
) -> str:

    # Infers the semantic type of a single column by examining
    # its pandas dtype, name, cardinality, and value patterns.
    # Returns one of the keys defined in SEMANTIC_TYPES.


    # -- Datetime --
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if _DT_NAME_HINTS.search(col_name):
        try:
            pd.to_datetime(series.dropna().head(20), errors="raise")
            return "datetime"
        except Exception:
            pass

    # -- Boolean --
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if n_unique == 2:
        vals = set(series.dropna().astype(str).str.lower().unique())
        if vals <= {"true", "false", "yes", "no", "si", "1", "0", "y", "n"}:
            return "boolean"

    # -- Identifier --
    if _ID_NAME_HINTS.search(col_name):
        return "id_like"
    if n_unique == n_rows and n_rows > 10 and not pd.api.types.is_numeric_dtype(series):
        return "id_like"

    # -- Numeric --
    if pd.api.types.is_numeric_dtype(series):
        if pd.api.types.is_integer_dtype(series) and n_unique <= max(20, n_rows * 0.05):
            return "numeric_discrete"
        return "numeric_continuous"

    # -- Categorical vs Free text --
    # pandas 3.0 uses StringDtype; is_string_dtype covers both object and StringDtype
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        ratio   = n_unique / max(n_rows, 1)
        avg_len = series.dropna().astype(str).str.len().mean()

        if ratio < 0.3 and n_unique <= 50:
            return "categorical"
        if avg_len > 40:
            return "text_free"
        if ratio < 0.6:
            return "categorical"

    return "unknown"

#  Global dataset metadata
def _build_meta(
    df: pd.DataFrame,
    columns: list[ColumnMeta],
    filename: str,
) -> dict:

    # Builds the metadata dictionary that will be passed to
    # analyzer.py, visualizer.py, and agent.py.

    total_missing = sum(c.n_missing for c in columns)
    total_cells   = df.size

    return {
        "filename": filename,
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "total_missing": total_missing,
        "pct_missing": round(total_missing / total_cells * 100, 2) if total_cells else 0,
        "memory_kb": round(df.memory_usage(deep=True).sum() / 1024, 1),
        "semantic_summary": {
            stype: [c.name for c in columns if c.semantic == stype]
            for stype in SEMANTIC_TYPES
            if any(c.semantic == stype for c in columns)
        },
    }

#  Public utility: readable schema
def schema_dataframe(dataset: LoadedDataset) -> pd.DataFrame:

    # Returns a human-readable DataFrame describing the schema.
    # Useful for displaying in Streamlit with st.dataframe().

    if not dataset or not dataset.columns:
        return pd.DataFrame()

    return pd.DataFrame([
        {
            "Column": c.name,
            "Pandas type": c.dtype_raw,
            "Semantic type": SEMANTIC_TYPES.get(c.semantic, c.semantic),
            "Unique values": c.n_unique,
            "Missing (%)": f"{c.pct_missing}%",
            "Examples": ", ".join(str(v) for v in c.sample_values[:3]),
        }
        for c in dataset.columns
    ])