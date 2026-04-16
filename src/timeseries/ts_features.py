"""
Time-Series Intelligence module for SSD telemetry data.

Adds time-aware features that capture how metrics evolve over time, exposing
degradation trends and sudden spikes that static per-row rules miss.

Features produced
-----------------
rolling_mean_latency   — rolling mean of latency_write_p99_ms (window rows)
rolling_std_latency    — rolling std  of latency_write_p99_ms
rolling_mean_iops      — rolling mean of total_iops
rolling_std_iops       — rolling std  of total_iops
latency_trend          — slope of latency over the rolling window (positive = rising)
latency_trend_flag     — True when slope > trend_threshold (sustained increase)
latency_spike          — True when current value > spike_multiplier × rolling mean
iops_spike             — True when current total_iops > spike_multiplier × rolling mean
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_WINDOW: int = 30          # rows (~5 min at 10-s sampling)
DEFAULT_MIN_PERIODS: int = 5
DEFAULT_TREND_THRESHOLD: float = 0.5   # ms per row — flag as trending up
DEFAULT_SPIKE_MULTIPLIER: float = 2.0  # current > 2× rolling mean → spike


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rolling_slope(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    """
    Compute an approximate linear slope over each rolling window using the
    least-squares formula:  slope = cov(x, y) / var(x)
    where x = [0, 1, …, n-1] for the actual slice length n.

    Returns a Series of the same length, NaN where fewer than min_periods
    observations are available.
    """
    def slope_of(window_vals: np.ndarray) -> float:
        n = len(window_vals)
        valid = np.sum(~np.isnan(window_vals))
        if valid < min_periods:
            return np.nan
        x = np.arange(n, dtype=float)
        x_mean = x.mean()
        x_var = ((x - x_mean) ** 2).sum()
        if x_var == 0:
            return np.nan
        y_mean = np.nanmean(window_vals)
        cov = np.nansum((x - x_mean) * (window_vals - y_mean))
        return cov / x_var

    return series.rolling(window=window, min_periods=min_periods).apply(
        slope_of, raw=True
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_ts_features(
    df: pd.DataFrame,
    window: int = DEFAULT_WINDOW,
    min_periods: int = DEFAULT_MIN_PERIODS,
    trend_threshold: float = DEFAULT_TREND_THRESHOLD,
    spike_multiplier: float = DEFAULT_SPIKE_MULTIPLIER,
    sort_by_time: bool = True,
) -> pd.DataFrame:
    """
    Add time-series intelligence features to a telemetry DataFrame.

    The DataFrame should have ``total_iops`` present (computed by
    ``feature_engineering.add_workload_behavior_features``).  If it is
    missing it is derived inline as ``read_iops + write_iops``.

    Parameters
    ----------
    df:
        Cleaned telemetry DataFrame, optionally with an ``event_ts`` column.
    window:
        Row-based rolling window size.
    min_periods:
        Minimum non-NaN observations required in the window.
    trend_threshold:
        Slope threshold (ms / row) above which ``latency_trend_flag`` is set.
    spike_multiplier:
        Multiple of the rolling mean above which a spike is declared.
    sort_by_time:
        If True and ``event_ts`` is present, sort by timestamp before computing
        rolling features, then restore the original index order.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with eight new columns appended.
    """
    out = df.copy()

    # Ensure total_iops exists
    if "total_iops" not in out.columns:
        out["total_iops"] = out["read_iops"] + out["write_iops"]

    # Sort by time if possible so rolling windows respect chronological order
    original_index = out.index.copy()
    if sort_by_time and "event_ts" in out.columns:
        ts = pd.to_datetime(out["event_ts"], errors="coerce", utc=True)
        out = out.assign(_sort_ts=ts).sort_values("_sort_ts").drop(columns="_sort_ts")

    roll_kwargs = dict(window=window, min_periods=min_periods)

    # Rolling statistics
    out["rolling_mean_latency"] = (
        out["latency_write_p99_ms"].rolling(**roll_kwargs).mean()
    )
    out["rolling_std_latency"] = (
        out["latency_write_p99_ms"].rolling(**roll_kwargs).std()
    )
    out["rolling_mean_iops"] = out["total_iops"].rolling(**roll_kwargs).mean()
    out["rolling_std_iops"] = out["total_iops"].rolling(**roll_kwargs).std()

    # Trend: slope of latency over the rolling window
    out["latency_trend"] = _rolling_slope(
        out["latency_write_p99_ms"], window=window, min_periods=min_periods
    )
    out["latency_trend_flag"] = out["latency_trend"] > trend_threshold

    # Spikes: current value > multiplier × rolling mean
    out["latency_spike"] = (
        out["latency_write_p99_ms"]
        > spike_multiplier * out["rolling_mean_latency"]
    )
    out["iops_spike"] = (
        out["total_iops"]
        > spike_multiplier * out["rolling_mean_iops"]
    )

    # Restore original row order
    out = out.reindex(original_index)

    return out


def get_ts_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a concise summary of time-series feature counts.

    Requires ``add_ts_features`` to have been called first.
    """
    required = {"latency_spike", "iops_spike", "latency_trend_flag"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns — call add_ts_features() first: {missing}")

    total = len(df)
    rows = [
        {
            "feature": "latency_spike",
            "count": int(df["latency_spike"].sum()),
            "pct": round(df["latency_spike"].mean() * 100, 2),
        },
        {
            "feature": "iops_spike",
            "count": int(df["iops_spike"].sum()),
            "pct": round(df["iops_spike"].mean() * 100, 2),
        },
        {
            "feature": "latency_trend_flag",
            "count": int(df["latency_trend_flag"].sum()),
            "pct": round(df["latency_trend_flag"].mean() * 100, 2),
        },
    ]
    return pd.DataFrame(rows)
