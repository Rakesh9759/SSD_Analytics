"""Statistical profiler for SSD telemetry DataFrames."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_PROFILE_COLS = [
    'read_iops', 'write_iops', 'latency_read_p99_ms', 'latency_write_p99_ms',
    'cpu_usage_pct', 'io_wait_pct', 'total_iops', 'io_mix',
    'write_amplification', 'throughput_per_iop', 'saturation_score',
    'cpu_to_io_ratio', 'io_wait_to_latency_ratio', 'burstiness',
]


def generate_data_profile(
    df: pd.DataFrame,
    profile_cols: list[str] | None = None,
    output_path: str | Path | None = None,
) -> dict:
    """Generate a statistical profile of the DataFrame.

    Returns
    -------
    dict with keys:
        distribution  — describe() table (mean, std, min, 50%, 95%, 99%, max)
        skew          — mean vs median comparison per column
        correlation   — correlation matrix of key telemetry columns
        group_profile — mean/median/count grouped by device_name, soc_model, storage_tier
    """
    cols = [c for c in (profile_cols or DEFAULT_PROFILE_COLS) if c in df.columns]

    # Distribution summary
    distribution = df[cols].describe(percentiles=[0.5, 0.95, 0.99]).T
    distribution = distribution[['mean', '50%', 'std', 'min', '95%', '99%', 'max']]

    # Skew detection
    skew = pd.DataFrame({
        'mean': df[cols].mean(),
        'median': df[cols].median(),
    })
    skew['mean_minus_median'] = skew['mean'] - skew['median']
    skew['mean_to_median_ratio'] = skew['mean'] / skew['median'].replace(0, np.nan)
    skew = skew.sort_values('mean_minus_median', ascending=False)

    # Correlation matrix
    corr_cols = [
        c for c in [
            'latency_read_p99_ms', 'latency_write_p99_ms', 'nvme_queue_depth',
            'io_wait_pct', 'cpu_usage_pct', 'memory_usage_pct', 'total_iops',
        ]
        if c in df.columns
    ]
    correlation = df[corr_cols].corr(numeric_only=True) if corr_cols else pd.DataFrame()

    # Grouped profiling
    group_keys = [k for k in ['device_name', 'soc_model', 'storage_tier'] if k in df.columns]
    group_metric_cols = [c for c in ['total_iops', 'latency_read_p99_ms', 'latency_write_p99_ms', 'write_amplification'] if c in df.columns]
    group_profile = pd.DataFrame()
    if group_keys and group_metric_cols:
        group_profile = (
            df.groupby(group_keys, dropna=False)[group_metric_cols]
            .agg(['mean', 'median', 'count'])
            .sort_values((group_metric_cols[0], 'mean'), ascending=False)
        )

    result = {
        'distribution': distribution,
        'skew': skew,
        'correlation': correlation,
        'group_profile': group_profile,
    }

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        distribution.to_csv(path)

    return result
