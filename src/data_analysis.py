"""Dataset profiling utilities built on Pandas.

This module performs data UNDERSTANDING only (profiling, quality checks).
It intentionally does not train or evaluate any Machine Learning model.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from .prompts import DATASET_CONTEXT_TEMPLATE, NO_DATASET_CONTEXT


class DatasetError(ValueError):
    """Raised when an uploaded CSV cannot be used for analysis."""


def load_csv(file) -> pd.DataFrame:
    """Load and lightly validate a CSV file-like object into a DataFrame."""
    try:
        df = pd.read_csv(file)
    except pd.errors.EmptyDataError as exc:
        raise DatasetError("File CSV kosong atau tidak memiliki data.") from exc
    except pd.errors.ParserError as exc:
        raise DatasetError("File tidak dapat dibaca sebagai CSV yang valid.") from exc
    except UnicodeDecodeError as exc:
        raise DatasetError("Encoding file tidak didukung. Gunakan CSV berformat UTF-8.") from exc

    if df.shape[1] == 0:
        raise DatasetError("Dataset tidak memiliki kolom.")
    if df.shape[0] == 0:
        raise DatasetError("Dataset tidak memiliki baris data.")
    return df


def _detect_id_columns(df: pd.DataFrame) -> list[str]:
    n = len(df)
    if n <= 1:
        return []
    id_cols = []
    for col in df.columns:
        name = str(col).lower()
        is_unique = df[col].nunique(dropna=True) == n
        name_hint = name == "id" or name.endswith("_id") or name.endswith("id") or name == "index"
        if is_unique and name_hint:
            id_cols.append(col)
    return id_cols


def _detect_outlier_columns(df: pd.DataFrame, numeric_cols: list[str]) -> list[dict]:
    outliers = []
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 4:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        flagged = series[(series < lower) | (series > upper)]
        if len(flagged) > 0:
            outliers.append(
                {
                    "column": col,
                    "count": int(len(flagged)),
                    "pct": round(len(flagged) / len(series) * 100, 2),
                }
            )
    return sorted(outliers, key=lambda o: o["count"], reverse=True)


def profile_dataset(df: pd.DataFrame, filename: str) -> dict:
    """Compute a structured profile of the dataset used both for the UI and as
    grounded context passed to the LLM (hallucination control)."""
    rows, cols = df.shape
    missing_series = df.isna().sum()
    missing_total = int(missing_series.sum())
    total_cells = max(rows * cols, 1)
    missing_pct = round(missing_total / total_cells * 100, 2)
    duplicate_rows = int(df.duplicated().sum())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    id_cols = _detect_id_columns(df)
    outlier_cols = _detect_outlier_columns(df, numeric_cols)

    column_details = []
    for col in df.columns:
        miss = int(missing_series[col])
        column_details.append(
            {
                "column": str(col),
                "dtype": str(df[col].dtype),
                "missing": miss,
                "missing_pct": round(miss / rows * 100, 2) if rows else 0.0,
                "unique": int(df[col].nunique(dropna=True)),
            }
        )

    missing_columns = [c["column"] for c in column_details if c["missing"] > 0]

    try:
        describe_df = df.describe(include="all").transpose().astype(str).replace("nan", "")
    except Exception:
        describe_df = pd.DataFrame()

    return {
        "filename": filename,
        "rows": rows,
        "columns": cols,
        "column_names": [str(c) for c in df.columns.tolist()],
        "column_details": column_details,
        "missing_total": missing_total,
        "missing_pct": missing_pct,
        "missing_columns": missing_columns,
        "duplicate_rows": duplicate_rows,
        "duplicate_pct": round(duplicate_rows / rows * 100, 2) if rows else 0.0,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "id_columns": id_cols,
        "outlier_columns": outlier_cols,
        "describe": describe_df,
    }


def profile_to_context_text(profile: dict | None) -> str:
    """Render the dataset profile as a compact, LLM-friendly context block."""
    if not profile:
        return NO_DATASET_CONTEXT

    missing_cols_str = ", ".join(profile["missing_columns"]) if profile["missing_columns"] else "Tidak ada"
    if profile["outlier_columns"]:
        outlier_str = ", ".join(
            f"{o['column']} ({o['count']} nilai, {o['pct']}%)" for o in profile["outlier_columns"]
        )
    else:
        outlier_str = "Tidak terdeteksi"
    id_str = ", ".join(profile["id_columns"]) if profile["id_columns"] else "Tidak terdeteksi"

    return DATASET_CONTEXT_TEMPLATE.format(
        filename=profile["filename"],
        rows=profile["rows"],
        columns=profile["columns"],
        column_names=", ".join(profile["column_names"]),
        missing_total=profile["missing_total"],
        missing_pct=profile["missing_pct"],
        missing_columns=missing_cols_str,
        duplicate_rows=profile["duplicate_rows"],
        numeric_columns=", ".join(profile["numeric_columns"]) if profile["numeric_columns"] else "Tidak ada",
        categorical_columns=", ".join(profile["categorical_columns"]) if profile["categorical_columns"] else "Tidak ada",
        id_columns=id_str,
        outlier_columns=outlier_str,
    )
