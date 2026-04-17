"""Reusable data quality validator for SSD telemetry DataFrames."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_quality.rules import (
    EXPECTED_COLS,
    NUMERIC_COLS,
    RANGE_RULES,
    VALID_SOC,
    VALID_STORAGE,
)


def run_data_quality_checks(
    df: pd.DataFrame,
    expected_cols: list[str] | None = None,
    numeric_cols: list[str] | None = None,
    range_rules: dict | None = None,
    valid_storage: set | None = None,
    valid_soc: set | None = None,
) -> dict:
    """Run all data quality checks and return a report dict.

    Returns
    -------
    dict with keys:
        df              — coerced copy of the input DataFrame
        schema          — {missing_cols, extra_cols, coercion_report}
        nulls           — {col_null_pct, row_null_pct_mean, device_id_null_pct}
        duplicates      — {exact, pk, near}
        range_issues    — list of {column, violations}
        cat_issues      — {soc_model_invalid, storage_tier_invalid}
        ts_stats        — {parsed_pct, future_ts, old_ts}
        dq_score_table  — DataFrame with component scores
        dq_score        — overall float score
    """
    expected_cols = expected_cols or EXPECTED_COLS
    numeric_cols = numeric_cols or NUMERIC_COLS
    range_rules = range_rules or RANGE_RULES
    valid_storage = valid_storage or VALID_STORAGE
    valid_soc = valid_soc or VALID_SOC

    out = df.copy()

    # --- Schema ---
    missing_cols = sorted(set(expected_cols) - set(out.columns))
    extra_cols = sorted(set(out.columns) - set(expected_cols))

    coercion_report: dict = {}
    for c in numeric_cols:
        if c not in out.columns:
            continue
        converted = pd.to_numeric(out[c], errors='coerce')
        coercion_report[c] = {
            'null_before': int(out[c].isna().sum()),
            'null_after': int(converted.isna().sum()),
            'bad_token_count': int(converted.isna().sum() - out[c].isna().sum()),
        }
        out[c] = converted

    if 'event_ts' in out.columns:
        out['event_ts'] = pd.to_datetime(out['event_ts'], errors='coerce', utc=True)

    # --- Nulls ---
    col_null_pct = (out.isna().mean() * 100).sort_values(ascending=False).round(2)
    device_id_null_pct = round(out['device_id'].isna().mean() * 100, 2) if 'device_id' in out.columns else None
    row_null_pct_mean = round(out.isna().mean(axis=1).mean() * 100, 2)

    # --- Duplicates ---
    exact_dups = int(out.duplicated().sum())
    pk_dups = int(out.duplicated(subset=['device_id', 'event_ts']).sum()) if {'device_id', 'event_ts'} <= set(out.columns) else 0

    near_dups = 0
    if {'device_id', 'event_ts'} <= set(out.columns):
        ts_ok = out.dropna(subset=['device_id', 'event_ts']).sort_values(['device_id', 'event_ts'])
        if not ts_ok.empty:
            delta = ts_ok.groupby('device_id')['event_ts'].diff().abs().dt.total_seconds()
            near_dups = int(((delta > 0) & (delta <= 5)).sum())

    # --- Range checks ---
    range_issues: list[dict] = []
    for col, (lo, hi) in range_rules.items():
        if col not in out.columns:
            continue
        bad = pd.Series(False, index=out.index)
        if lo is not None:
            bad = bad | (out[col] < lo)
        if hi is not None:
            bad = bad | (out[col] > hi)
        range_issues.append({'column': col, 'violations': int(bad.fillna(False).sum())})

    # --- Categorical checks ---
    cat_issues: dict = {}
    if 'soc_model' in out.columns:
        cat_issues['soc_model_invalid'] = int((~out['soc_model'].isin(valid_soc)).fillna(True).sum())
    if 'storage_tier' in out.columns:
        cat_issues['storage_tier_invalid'] = int((~out['storage_tier'].isin(valid_storage)).fillna(True).sum())

    # --- Timestamp checks ---
    ts_stats: dict = {}
    if 'event_ts' in out.columns:
        ts_stats['parsed_pct'] = round(out['event_ts'].notna().mean() * 100, 2)
        now_utc = pd.Timestamp.now(tz='UTC')
        ts_stats['future_ts'] = int((out['event_ts'] > now_utc).fillna(False).sum())
        ts_stats['old_ts'] = int((out['event_ts'] < pd.Timestamp('2000-01-01', tz='UTC')).fillna(False).sum())

    # --- DQ score ---
    total_range_violations = sum(r['violations'] for r in range_issues)
    total_cat_violations = sum(cat_issues.values())

    completeness = (1 - out.isna().mean().mean()) * 100
    validity = 100 - (total_range_violations / max(len(out) * max(len(range_issues), 1), 1) * 100)
    uniqueness = (1 - (exact_dups / max(len(out), 1))) * 100
    consistency = 100 - (total_cat_violations / max(2 * len(out), 1) * 100)
    dq_score = (completeness + validity + uniqueness + consistency) / 4

    dq_score_table = pd.DataFrame([
        {'component': 'completeness',     'score': round(completeness, 2)},
        {'component': 'validity',         'score': round(validity, 2)},
        {'component': 'uniqueness',       'score': round(uniqueness, 2)},
        {'component': 'consistency',      'score': round(consistency, 2)},
        {'component': 'overall_dq_score', 'score': round(dq_score, 2)},
    ])

    return {
        'df': out,
        'schema': {
            'missing_cols': missing_cols,
            'extra_cols': extra_cols,
            'coercion_report': coercion_report,
        },
        'nulls': {
            'col_null_pct': col_null_pct,
            'row_null_pct_mean': row_null_pct_mean,
            'device_id_null_pct': device_id_null_pct,
        },
        'duplicates': {
            'exact': exact_dups,
            'pk': pk_dups,
            'near': near_dups,
        },
        'range_issues': range_issues,
        'cat_issues': cat_issues,
        'ts_stats': ts_stats,
        'dq_score_table': dq_score_table,
        'dq_score': round(dq_score, 2),
    }
