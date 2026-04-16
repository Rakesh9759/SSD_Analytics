"""
Root Cause Analysis Engine for SSD telemetry anomalies.

Applies a priority-ordered rule engine to classify *why* a row is anomalous,
rather than just *whether* it is.  Outputs a ``root_cause_label`` and a
``confidence_score`` (0–1) for each row.

Rules (evaluated in priority order):
    1. disk_bottleneck    — latency high AND io_wait high
    2. cpu_bottleneck     — latency high AND cpu high
    3. memory_pressure    — latency high AND memory high
    4. suspicious_unknown — latency high but all other signals normal
    5. normal             — latency within threshold

Confidence score:
    Proportion of the triggering signals that exceeded their thresholds.
    Ranges from 0 (no signals fired) to 1 (all checked signals are elevated).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Default thresholds — can be overridden by callers
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLDS: dict[str, float] = {
    "latency_write_p99_ms": 80.0,
    "io_wait_pct": 20.0,
    "cpu_usage_pct": 85.0,
    "memory_usage_pct": 85.0,
}

# Labels
LABEL_DISK = "disk_bottleneck"
LABEL_CPU = "cpu_bottleneck"
LABEL_MEMORY = "memory_pressure"
LABEL_SUSPICIOUS = "suspicious_unknown"
LABEL_NORMAL = "normal"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_root_cause(
    df: pd.DataFrame,
    thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Classify the root cause of each row and return the DataFrame with two
    new columns appended:

    * ``root_cause_label``  — one of the five label constants above.
    * ``confidence_score``  — float in [0, 1]; how strongly the primary rule fired.

    Parameters
    ----------
    df:
        DataFrame that must contain at least the columns referenced by
        ``thresholds``.  Additional engineered features (from
        ``feature_engineering.build_features``) are used when present.
    thresholds:
        Override any of the default thresholds.  Keys must match column names.

    Returns
    -------
    pd.DataFrame
        A copy of ``df`` with ``root_cause_label`` and ``confidence_score``
        appended.
    """
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    out = df.copy()

    lat_high = out["latency_write_p99_ms"] > t["latency_write_p99_ms"]
    io_high = out["io_wait_pct"] > t["io_wait_pct"]
    cpu_high = out["cpu_usage_pct"] > t["cpu_usage_pct"]
    mem_high = out["memory_usage_pct"] > t["memory_usage_pct"]

    # -----------------------------------------------------------------------
    # Labels — priority order (first match wins via np.select)
    # -----------------------------------------------------------------------
    conditions = [
        lat_high & io_high,          # disk_bottleneck
        lat_high & cpu_high,         # cpu_bottleneck
        lat_high & mem_high,         # memory_pressure
        lat_high,                    # suspicious_unknown (latency alone)
    ]
    choices = [LABEL_DISK, LABEL_CPU, LABEL_MEMORY, LABEL_SUSPICIOUS]
    out["root_cause_label"] = np.select(conditions, choices, default=LABEL_NORMAL)

    # -----------------------------------------------------------------------
    # Confidence score
    # -----------------------------------------------------------------------
    # For each row, count how many of the four signals are above threshold
    # and normalise by 4 (the total number of monitored signals).
    signal_matrix = pd.DataFrame(
        {
            "lat_high": lat_high.astype(int),
            "io_high": io_high.astype(int),
            "cpu_high": cpu_high.astype(int),
            "mem_high": mem_high.astype(int),
        }
    )
    out["confidence_score"] = signal_matrix.sum(axis=1) / len(signal_matrix.columns)

    return out


def get_root_cause_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate root cause labels into a summary table.

    Requires ``root_cause_label`` and ``confidence_score`` to be present
    (i.e. ``classify_root_cause`` must have been called first).

    Returns a DataFrame with columns:
        root_cause_label, count, pct, mean_confidence
    """
    if "root_cause_label" not in df.columns:
        raise ValueError(
            "root_cause_label column not found — call classify_root_cause() first."
        )

    total = len(df)
    summary = (
        df.groupby("root_cause_label", sort=False)
        .agg(
            count=("root_cause_label", "size"),
            mean_confidence=("confidence_score", "mean"),
        )
        .reset_index()
    )
    summary["pct"] = (summary["count"] / total * 100).round(2)
    summary["mean_confidence"] = summary["mean_confidence"].round(3)
    summary = summary.sort_values("count", ascending=False).reset_index(drop=True)
    return summary
