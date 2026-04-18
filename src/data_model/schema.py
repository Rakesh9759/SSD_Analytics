"""Star schema definitions and mart table builders for SSD telemetry.

Tables
------
Fact:
    fact_device_metrics  — one row per device per event timestamp

Dimensions:
    dim_device  — one row per device (device_id, name, storage_tier, soc_model)
    dim_time    — calendar/time attributes derived from event_ts

Marts (pre-aggregated, ready for dashboards):
    mart_device_overview    — per-device summary (mean latency, iops, anomaly rate)
    mart_anomaly_timeline   — anomaly flags over time (hourly buckets)
    mart_root_cause         — root cause label counts and mean confidence
"""

from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# Fact table
# ---------------------------------------------------------------------------

FACT_COLS = [
    "run_id", "event_ts", "device_id",
    "read_iops", "write_iops", "total_iops",
    "read_throughput_mb_s", "write_throughput_mb_s",
    "latency_read_p99_ms", "latency_write_p99_ms",
    "utilization_pct", "nvme_queue_depth",
    "cpu_usage_pct", "memory_usage_pct", "io_wait_pct",
    "host_writes_mb", "nand_writes_mb", "thermal_throttling_events",
    "write_amplification", "throughput_per_iop", "saturation_score",
    "io_mix", "cpu_to_io_ratio", "io_wait_to_latency_ratio", "burstiness",
    "anomaly_score", "if_anomaly", "mz_anomaly", "detected_anomaly",
    "root_cause_label", "confidence_score",
    "latency_trend", "latency_trend_flag", "latency_spike", "iops_spike",
]


def build_fact_device_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Select fact columns present in df; drop rows missing device_id or event_ts."""
    cols = [c for c in FACT_COLS if c in df.columns]
    fact = df[cols].dropna(subset=["device_id", "event_ts"]).copy()
    fact = fact.sort_values(["device_id", "event_ts"]).reset_index(drop=True)
    return fact


# ---------------------------------------------------------------------------
# Dimension tables
# ---------------------------------------------------------------------------

def build_dim_device(df: pd.DataFrame) -> pd.DataFrame:
    """One row per device with its latest metadata."""
    cols = [c for c in ["device_id", "device_name", "storage_tier", "soc_model"] if c in df.columns]
    dim = (
        df[cols]
        .dropna(subset=["device_id"])
        .drop_duplicates(subset=["device_id"])
        .sort_values("device_id")
        .reset_index(drop=True)
    )
    return dim


def build_dim_time(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar attributes derived from event_ts; one row per unique timestamp."""
    if "event_ts" not in df.columns:
        return pd.DataFrame()
    ts = pd.to_datetime(df["event_ts"], errors="coerce", utc=True).dropna().unique()
    ts_series = pd.Series(sorted(ts), name="event_ts")
    dim = pd.DataFrame({"event_ts": ts_series})
    dim["date"]       = dim["event_ts"].dt.date
    dim["hour"]       = dim["event_ts"].dt.hour
    dim["day_of_week"] = dim["event_ts"].dt.day_name()
    dim["week"]       = dim["event_ts"].dt.isocalendar().week.astype(int)
    dim["month"]      = dim["event_ts"].dt.month
    dim["year"]       = dim["event_ts"].dt.year
    return dim.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Mart tables
# ---------------------------------------------------------------------------

def build_mart_device_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Per-device summary: mean latency, iops, utilization, anomaly rate."""
    group_cols = [c for c in ["device_id", "device_name", "storage_tier", "soc_model"] if c in df.columns]
    agg: dict = {}
    for col in ["latency_read_p99_ms", "latency_write_p99_ms", "total_iops",
                "cpu_usage_pct", "io_wait_pct", "utilization_pct",
                "write_amplification", "saturation_score", "anomaly_score"]:
        if col in df.columns:
            agg[col] = "mean"
    if "detected_anomaly" in df.columns:
        agg["detected_anomaly"] = "mean"
    if "run_id" in df.columns:
        agg["run_id"] = "count"

    mart = (
        df.groupby(group_cols, dropna=False)
        .agg(agg)
        .rename(columns={"detected_anomaly": "anomaly_rate", "run_id": "row_count"})
        .round(4)
        .reset_index()
        .sort_values("anomaly_rate", ascending=False)
    )
    return mart


def build_mart_anomaly_timeline(df: pd.DataFrame) -> pd.DataFrame:
    """Hourly anomaly counts and rates across the fleet."""
    if "event_ts" not in df.columns:
        return pd.DataFrame()
    t = df.copy()
    t["event_ts"] = pd.to_datetime(t["event_ts"], errors="coerce", utc=True)
    t = t.dropna(subset=["event_ts"])
    t["hour_bucket"] = t["event_ts"].dt.floor("h")

    agg: dict = {"run_id": "count"}
    for col in ["detected_anomaly", "if_anomaly", "mz_anomaly", "is_anomaly"]:
        if col in t.columns:
            agg[col] = "sum"

    mart = (
        t.groupby("hour_bucket")
        .agg(agg)
        .rename(columns={"run_id": "total_rows"})
        .reset_index()
        .sort_values("hour_bucket")
    )
    if "detected_anomaly" in mart.columns and "total_rows" in mart.columns:
        mart["anomaly_rate"] = (mart["detected_anomaly"] / mart["total_rows"]).round(4)
    return mart


def build_mart_root_cause(df: pd.DataFrame) -> pd.DataFrame:
    """Root cause label counts, anomaly rate, and mean confidence."""
    if "root_cause_label" not in df.columns:
        return pd.DataFrame()
    agg: dict = {"run_id": "count"}
    if "confidence_score" in df.columns:
        agg["confidence_score"] = "mean"
    if "detected_anomaly" in df.columns:
        agg["detected_anomaly"] = "mean"
    if "anomaly_score" in df.columns:
        agg["anomaly_score"] = "mean"

    mart = (
        df.groupby("root_cause_label")
        .agg(agg)
        .rename(columns={"run_id": "row_count", "detected_anomaly": "anomaly_rate"})
        .round(4)
        .reset_index()
        .sort_values("row_count", ascending=False)
    )
    return mart
