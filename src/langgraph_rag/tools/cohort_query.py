# src/langgraph_rag/tools/cohort_query.py

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Any, List, Tuple

import pandas as pd

from langgraph_rag.config import IngestionConfig


@lru_cache(maxsize=16)
def _load_table(path: str) -> pd.DataFrame:
    """Load a TSV file once and cache it."""
    return pd.read_csv(path, sep="\t")


def load_tables_from_config(config: IngestionConfig) -> Dict[str, pd.DataFrame]:
    """
    Load all tables defined in the ingestion config.
    Returns a mapping: table_name -> DataFrame
    """
    tables: Dict[str, pd.DataFrame] = {}
    for t in config.tables:
        tables[t.name] = _load_table(t.path)
    return tables


def _apply_filter(df: pd.DataFrame, filter_spec: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply a simple AND filter over columns.

    For each key, value in filter_spec:
      - if value is a list: df[col].isin(value)
      - else: case-insensitive substring match on df[col].astype(str)
    """
    if not filter_spec:
        return df

    mask = pd.Series(True, index=df.index)
    for col, val in filter_spec.items():
        if col not in df.columns:
            # Column missing: drop everything for this filter
            mask &= False
            continue

        series = df[col].astype(str)
        if isinstance(val, list):
            mask &= series.isin([str(v) for v in val])
        else:
            mask &= series.str.contains(str(val), case=False, na=False)

    return df[mask]


def execute_cohort_filter_step(
    step: Dict[str, Any],
    tables: Dict[str, pd.DataFrame],
    context: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Execute a 'cohort_sql' step:
      - Pick the first table in step['tables_used'].
      - Apply step['filter'].
      - Store the resulting cohort in context['current_cohort'].
    """
    tables_used: List[str] = step.get("tables_used") or []
    filter_spec: Dict[str, Any] = step.get("filter") or {}

    # Basic robustness: if planner used "clinical" but our table is named "patients", map it.
    table_name = tables_used[0] if tables_used else None
    if table_name == "clinical" and "patients" in tables:
        table_name = "patients"

    if not table_name or table_name not in tables:
        return {
            "step_id": step.get("step_id"),
            "name": step.get("name"),
            "tool": step.get("tool"),
            "status": "failed",
            "error": f"Unknown table: {table_name}",
        }, context

    df = tables[table_name]
    filtered = _apply_filter(df, filter_spec)

    patient_id_cols = [c for c in df.columns if "Patient ID" in c or c.lower() == "patient_id"]
    if patient_id_cols:
        pid_col = patient_id_cols[0]
        n_patients = filtered[pid_col].nunique()
        sample_ids = filtered[pid_col].dropna().astype(str).head(5).tolist()
    else:
        pid_col = None
        n_patients = None
        sample_ids = []

    # Update context
    context["current_cohort"] = filtered
    context["current_cohort_table"] = table_name
    context["current_cohort_patient_id_col"] = pid_col

    result: Dict[str, Any] = {
        "step_id": step.get("step_id"),
        "name": step.get("name"),
        "tool": step.get("tool"),
        "status": "success",
        "table_used": table_name,
        "filter_applied": filter_spec,
        "row_count": int(filtered.shape[0]),
        "patient_count": int(n_patients) if n_patients is not None else None,
        "patient_id_column": pid_col,
        "sample_patient_ids": sample_ids,
    }
    return result, context


def execute_feature_descriptives_step(
    step: Dict[str, Any],
    tables: Dict[str, pd.DataFrame],
    context: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Execute a 'feature_descriptives' step:
      - Use context['current_cohort'] if present, else fall back to the table in tables_used.
      - Compute mean/median/std/IQR for the specified feature.
    """
    feature = step.get("feature")
    tables_used: List[str] = step.get("tables_used") or []

    cohort_df = context.get("current_cohort")
    source = "current_cohort"

    if cohort_df is None:
        # Fall back to raw table if no cohort yet
        table_name = tables_used[0] if tables_used else None
        if table_name == "clinical" and "patients" in tables:
            table_name = "patients"

        if not table_name or table_name not in tables:
            return {
                "step_id": step.get("step_id"),
                "name": step.get("name"),
                "tool": step.get("tool"),
                "status": "failed",
                "error": f"No cohort and unknown table: {table_name}",
            }, context

        cohort_df = tables[table_name]
        source = f"table:{table_name}"

    if not feature or feature not in cohort_df.columns:
        return {
            "step_id": step.get("step_id"),
            "name": step.get("name"),
            "tool": step.get("tool"),
            "status": "failed",
            "source": source,
            "error": f"Feature column not found: {feature}",
        }, context

    series = pd.to_numeric(cohort_df[feature], errors="coerce").dropna()
    if series.empty:
        return {
            "step_id": step.get("step_id"),
            "name": step.get("name"),
            "tool": step.get("tool"),
            "status": "failed",
            "source": source,
            "error": f"No numeric values available for feature: {feature}",
        }, context

    desc = series.describe()  # count, mean, std, min, 25%, 50%, 75%, max
    q1 = float(desc["25%"])
    q3 = float(desc["75%"])

    result: Dict[str, Any] = {
        "step_id": step.get("step_id"),
        "name": step.get("name"),
        "tool": step.get("tool"),
        "status": "success",
        "source": source,
        "feature": feature,
        "metrics": {
            "count": int(desc["count"]),
            "mean": float(desc["mean"]),
            "std": float(desc["std"]) if not pd.isna(desc["std"]) else None,
            "min": float(desc["min"]),
            "max": float(desc["max"]),
            "median": float(desc["50%"]),
            "iqr": [q1, q3],
        },
    }
    return result, context
