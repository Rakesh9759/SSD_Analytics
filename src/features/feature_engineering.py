"""
Feature engineering for SSD telemetry data.

Produces derived columns that downstream anomaly detection and root cause
analysis depend on.

"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core static features
# ---------------------------------------------------------------------------

def add_disk_efficiency_features(telemetry_df: pd.DataFrame) -> pd.DataFrame:
    """
    Write Amplification Factor (WAF) and throughput-per-IOP.

    WAF > 1 means the NAND is doing more work than the host requested —
    a meaningful wear and efficiency signal.
    """
    telemetry_df = telemetry_df.copy()

    telemetry_df["write_amplification"] = (
        telemetry_df["nand_writes_mb"] / telemetry_df["host_writes_mb"]
    ).replace([np.inf, -np.inf], np.nan)

    total_throughput = (
        telemetry_df["read_throughput_mb_s"] + telemetry_df["write_throughput_mb_s"]
    )
    total_iops = telemetry_df["read_iops"] + telemetry_df["write_iops"]
    telemetry_df["throughput_per_iop"] = (
        total_throughput / total_iops
    ).replace([np.inf, -np.inf], np.nan)

    return telemetry_df


def add_saturation_features(telemetry_df: pd.DataFrame) -> pd.DataFrame:
    """
    Saturation score: queue depth × write latency.

    Combines two independent stress signals into a single scalar that
    amplifies when both are elevated simultaneously.
    """
    telemetry_df = telemetry_df.copy()

    telemetry_df["saturation_score"] = (
        telemetry_df["nvme_queue_depth"] * telemetry_df["latency_write_p99_ms"]
    )

    return telemetry_df


def add_workload_behavior_features(telemetry_df: pd.DataFrame) -> pd.DataFrame:
    """
    IO mix and system-correlation ratios.

    io_mix = read share of total IOPS (0 = write-only, 1 = read-only).
    cpu_to_io_ratio and io_wait_to_latency_ratio expose cross-resource
    relationships used by the root cause engine.
    """
    telemetry_df = telemetry_df.copy()

    telemetry_df["total_iops"] = (
        telemetry_df["read_iops"] + telemetry_df["write_iops"]
    )
    telemetry_df["io_mix"] = (
        telemetry_df["read_iops"] / telemetry_df["total_iops"]
    ).replace([np.inf, -np.inf], np.nan)

    telemetry_df["cpu_to_io_ratio"] = (
        telemetry_df["cpu_usage_pct"] / telemetry_df["io_wait_pct"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)

    telemetry_df["io_wait_to_latency_ratio"] = (
        telemetry_df["io_wait_pct"] / telemetry_df["latency_write_p99_ms"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)

    return telemetry_df


# ---------------------------------------------------------------------------
# Rolling / time-aware features 
# ---------------------------------------------------------------------------

def add_rolling_features(
    telemetry_df: pd.DataFrame,
    window: int = 30,
    min_periods: int = 5,
) -> pd.DataFrame:
    """
    Rolling mean/std for latency and IOPS, plus a burstiness indicator.

    The DataFrame must already have a parsed datetime index or an
    ``event_ts`` column.  If neither is present the features are added as
    NaN columns so the rest of the pipeline still runs.

    Parameters
    ----------
    window:
        Number of rows for the rolling window (default 30 ≈ 5-min sample).
    min_periods:
        Minimum observations required to produce a non-NaN result.
    """
    telemetry_df = telemetry_df.copy()

    has_ts = isinstance(telemetry_df.index, pd.DatetimeIndex)

    if not has_ts and "event_ts" in telemetry_df.columns:
        telemetry_df = telemetry_df.sort_values("event_ts")

    roll_kwargs = dict(window=window, min_periods=min_periods)

    telemetry_df["rolling_mean_latency"] = (
        telemetry_df["latency_write_p99_ms"].rolling(**roll_kwargs).mean()
    )
    telemetry_df["rolling_std_latency"] = (
        telemetry_df["latency_write_p99_ms"].rolling(**roll_kwargs).std()
    )
    telemetry_df["rolling_mean_iops"] = (
        telemetry_df["total_iops"].rolling(**roll_kwargs).mean()
    )
    telemetry_df["rolling_std_iops"] = (
        telemetry_df["total_iops"].rolling(**roll_kwargs).std()
    )
    telemetry_df["burstiness"] = (
        telemetry_df["rolling_std_iops"]
        / telemetry_df["rolling_mean_iops"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)

    return telemetry_df


def build_features(
    telemetry_df: pd.DataFrame,
    rolling_window: int = 30,
    rolling_min_periods: int = 5,
) -> pd.DataFrame:
    """
    Apply all feature engineering steps in the correct dependency order.

    1. Disk efficiency  (write_amplification, throughput_per_iop)
    2. Saturation       (saturation_score)
    3. Workload         (total_iops, io_mix, cpu_to_io_ratio, io_wait_to_latency_ratio)
    4. Rolling          (rolling_mean_latency, rolling_std_iops, burstiness, …)

    Parameters
    ----------
    telemetry_df:
        Raw or cleaned telemetry DataFrame.  Numeric columns are expected
        to already be in their correct dtypes.
    rolling_window:
        Row-based rolling window size (passed to ``add_rolling_features``).
    rolling_min_periods:
        Minimum periods for rolling calculations.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with all original columns plus engineered features.
    """
    telemetry_df = add_disk_efficiency_features(telemetry_df)
    telemetry_df = add_saturation_features(telemetry_df)
    telemetry_df = add_workload_behavior_features(telemetry_df)
    telemetry_df = add_rolling_features(
        telemetry_df,
        window=rolling_window,
        min_periods=rolling_min_periods,
    )
    return telemetry_df
