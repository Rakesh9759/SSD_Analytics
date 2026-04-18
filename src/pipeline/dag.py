"""Airflow-style pipeline orchestration for SSD Analytics.

Stages
------
generate_data -> corrupt_data -> validate_data -> clean_data
-> feature_engineering -> anomaly_detection -> root_cause

The module is intentionally lightweight and framework-agnostic: each stage is a
plain Python function, while run_pipeline() executes them in a DAG-like order
and persists key artifacts to data/generated.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from generate_raw_telemetry import _generate_metric_rows, _generate_run_rows
from generate_corrupted_telemetry import corrupt_data, inject_anomaly_scenarios
from src.anomaly.isolation_forest_detector import run_all_detectors
from src.data_quality.validator import run_data_quality_checks
from src.features.feature_engineering import build_features
from src.root_cause.root_cause_engine import classify_root_cause


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for the orchestration DAG."""

    rows_per_group: int = 100
    runs_per_group: int = 2
    seed: int = 42
    output_dir: Path = Path("data/generated")
    contamination: float = 0.1
    zscore_threshold: float = 3.0


def stage_generate_data(config: PipelineConfig, rng: np.random.Generator) -> pd.DataFrame:
    """Generate base telemetry by composing the raw telemetry generator helpers."""
    runs_df = _generate_run_rows(rng=rng, runs_per_group=config.runs_per_group)
    metrics_df = _generate_metric_rows(
        rng=rng,
        runs_df=runs_df,
        rows_per_group=config.rows_per_group,
    )
    return metrics_df


def stage_corrupt_data(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Inject realistic anomaly scenarios first, then apply corruption patterns."""
    with_anomalies = inject_anomaly_scenarios(df, rng=rng)
    return corrupt_data(with_anomalies, rng=rng)


def stage_validate_data(df: pd.DataFrame) -> dict[str, Any]:
    """Run reusable data quality validation checks."""
    return run_data_quality_checks(df)


def stage_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic cleaning so downstream models receive valid numeric input."""
    out = df.copy()

    if "event_ts" in out.columns:
        out["event_ts"] = pd.to_datetime(out["event_ts"], errors="coerce", utc=True)

    numeric_like_cols = [
        "read_iops",
        "write_iops",
        "read_throughput_mb_s",
        "write_throughput_mb_s",
        "latency_read_p99_ms",
        "latency_write_p99_ms",
        "utilization_pct",
        "nvme_queue_depth",
        "host_writes_mb",
        "nand_writes_mb",
        "thermal_throttling_events",
        "cpu_usage_pct",
        "memory_usage_pct",
        "io_wait_pct",
    ]
    for col in numeric_like_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # Keep one row per device/timestamp pair and drop rows that cannot be modeled.
    subset = [c for c in ["device_id", "event_ts"] if c in out.columns]
    if subset:
        out = out.drop_duplicates(subset=subset)
    out = out.dropna(subset=[c for c in ["event_ts", "latency_write_p99_ms"] if c in out.columns])

    # Clamp obvious out-of-range values injected by corruption stage.
    clamp_rules = {
        "utilization_pct": (0.0, 100.0),
        "cpu_usage_pct": (0.0, 100.0),
        "memory_usage_pct": (0.0, 100.0),
        "io_wait_pct": (0.0, 100.0),
        "latency_read_p99_ms": (0.0, None),
        "latency_write_p99_ms": (0.0, None),
        "read_iops": (0.0, None),
        "write_iops": (0.0, None),
        "nvme_queue_depth": (0.0, None),
        "host_writes_mb": (0.0, None),
        "nand_writes_mb": (0.0, None),
    }
    for col, (lo, hi) in clamp_rules.items():
        if col in out.columns:
            if lo is not None:
                out[col] = out[col].clip(lower=lo)
            if hi is not None:
                out[col] = out[col].clip(upper=hi)

    if "storage_tier" in out.columns:
        out["storage_tier"] = out["storage_tier"].replace("", np.nan).fillna("unknown")
    if "soc_model" in out.columns:
        out["soc_model"] = out["soc_model"].replace("", np.nan).fillna("unknown")
    if "device_name" in out.columns:
        out["device_name"] = out["device_name"].replace("", np.nan).fillna("unknown")

    return out.reset_index(drop=True)


def stage_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Apply engineered telemetry features used by detectors and RCA."""
    return build_features(df)


def stage_anomaly_detection(
    df: pd.DataFrame,
    contamination: float,
    zscore_threshold: float,
) -> pd.DataFrame:
    """Apply IF + multivariate z-score and add detector outputs."""
    return run_all_detectors(
        df,
        contamination=contamination,
        zscore_threshold=zscore_threshold,
    )


def stage_root_cause(df: pd.DataFrame) -> pd.DataFrame:
    """Classify each row with root-cause label and confidence score."""
    return classify_root_cause(df)


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    """Execute the full DAG and return stage metrics + output paths."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.seed)

    stage_stats: list[dict[str, Any]] = []

    def track(stage_name: str, start_time: float, row_count: int | None) -> None:
        stage_stats.append(
            {
                "stage": stage_name,
                "seconds": round(perf_counter() - start_time, 3),
                "rows": row_count,
            }
        )

    t0 = perf_counter()
    raw_df = stage_generate_data(config=config, rng=rng)
    track("generate_data", t0, len(raw_df))

    t0 = perf_counter()
    corrupted_df = stage_corrupt_data(raw_df, rng=rng)
    track("corrupt_data", t0, len(corrupted_df))

    t0 = perf_counter()
    dq_report = stage_validate_data(corrupted_df)
    track("validate_data", t0, len(dq_report["df"]))

    t0 = perf_counter()
    cleaned_df = stage_clean_data(dq_report["df"])
    track("clean_data", t0, len(cleaned_df))

    t0 = perf_counter()
    featured_df = stage_feature_engineering(cleaned_df)
    track("feature_engineering", t0, len(featured_df))

    t0 = perf_counter()
    detected_df = stage_anomaly_detection(
        featured_df,
        contamination=config.contamination,
        zscore_threshold=config.zscore_threshold,
    )
    track("anomaly_detection", t0, len(detected_df))

    t0 = perf_counter()
    final_df = stage_root_cause(detected_df)
    track("root_cause", t0, len(final_df))

    output_paths = {
        "raw": config.output_dir / "pipeline_raw_dataset.csv",
        "corrupted": config.output_dir / "pipeline_corrupted_dataset.csv",
        "cleaned": config.output_dir / "pipeline_cleaned_dataset.csv",
        "output": config.output_dir / "pipeline_root_cause_output.csv",
        "dq_score_table": config.output_dir / "pipeline_dq_score_table.csv",
        "stage_metrics": config.output_dir / "pipeline_stage_metrics.csv",
    }

    _save_csv(raw_df, output_paths["raw"])
    _save_csv(corrupted_df, output_paths["corrupted"])
    _save_csv(cleaned_df, output_paths["cleaned"])
    _save_csv(final_df, output_paths["output"])
    dq_report["dq_score_table"].to_csv(output_paths["dq_score_table"], index=False)
    pd.DataFrame(stage_stats).to_csv(output_paths["stage_metrics"], index=False)

    return {
        "config": config,
        "stage_stats": stage_stats,
        "dq_score": dq_report["dq_score"],
        "output_paths": output_paths,
        "final_rows": len(final_df),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SSD Analytics orchestration DAG")
    parser.add_argument("--rows-per-group", type=int, default=100)
    parser.add_argument("--runs-per-group", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("data/generated"))
    parser.add_argument("--contamination", type=float, default=0.1)
    parser.add_argument("--zscore-threshold", type=float, default=3.0)
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    config = PipelineConfig(
        rows_per_group=args.rows_per_group,
        runs_per_group=args.runs_per_group,
        seed=args.seed,
        output_dir=args.output_dir,
        contamination=args.contamination,
        zscore_threshold=args.zscore_threshold,
    )

    result = run_pipeline(config)

    print("Pipeline completed")
    print(f"Final rows: {result['final_rows']}")
    print(f"DQ score: {result['dq_score']}")
    print("Stage metrics:")
    for row in result["stage_stats"]:
        print(f"  - {row['stage']}: {row['rows']} rows, {row['seconds']}s")
    print("Outputs:")
    for name, path in result["output_paths"].items():
        print(f"  - {name}: {path}")


if __name__ == "__main__":
    main()
