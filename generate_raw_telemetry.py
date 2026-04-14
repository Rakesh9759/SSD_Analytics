from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd


APPLE_DEVICE_MAP = {
    "iPhone": ["iPhone 14 Pro", "iPhone 15", "iPhone 15 Pro"],
    "Mac": ["MacBook Pro M2", "MacBook Air M2", "Mac Mini M2"],
    "iPad": ["iPad Pro M2", "iPad Air M1"],
    "Apple Watch": ["Apple Watch Series 8", "Apple Watch Ultra"],
}

STORAGE_TIERS = ["128GB", "256GB", "512GB", "1TB"]
SOC_MODELS = ["A15", "A16", "A17", "M1", "M2"]

GROUP_PROFILES = {
    "iPhone": {"queue_depth": 10, "host_writes_mean": 320, "thermal_lambda": 2},
    "Mac": {"queue_depth": 20, "host_writes_mean": 1300, "thermal_lambda": 3},
    "iPad": {"queue_depth": 11, "host_writes_mean": 450, "thermal_lambda": 1},
    "Apple Watch": {"queue_depth": 4, "host_writes_mean": 90, "thermal_lambda": 4},
}

def _generate_run_rows(rng: np.random.Generator, runs_per_group: int) -> pd.DataFrame:
    rows = []
    now = datetime.now(timezone.utc)

    for group in GROUP_PROFILES:
        for idx in range(runs_per_group):
            start = now - timedelta(days=(runs_per_group - idx) * 2)

            rows.append(
                {
                    "run_id": str(uuid4()),
                    "device_group": group,
                    "firmware_version": f"16.{idx+4}",
                    "ssd_model": f"APSSD-{rng.integers(100,999)}",
                    "test_type": ["endurance", "performance", "thermal"][idx % 3],
                    "started_at": start.isoformat(),
                    "ended_at": (start + timedelta(hours=4)).isoformat(),
                    "status": "completed",
                }
            )

    return pd.DataFrame(rows)


def _generate_metric_rows(
    rng: np.random.Generator,
    runs_df: pd.DataFrame,
    rows_per_group: int,
) -> pd.DataFrame:
    out = []

    for _, run in runs_df.iterrows():
        profile = GROUP_PROFILES[run["device_group"]]
        start = pd.Timestamp(run["started_at"])

        for i in range(rows_per_group):

            device_name = rng.choice(APPLE_DEVICE_MAP[run["device_group"]])
            storage_tier = rng.choice(STORAGE_TIERS)
            soc_model = rng.choice(SOC_MODELS)

            read_iops = float(max(10, rng.normal(800, 200)))
            write_iops = float(max(10, rng.normal(600, 150)))

            read_tp = float(max(1.0, rng.normal(120, 30)))
            write_tp = float(max(1.0, rng.normal(90, 25)))

            latency_read = float(max(1.0, rng.normal(20, 5)))
            latency_write = float(max(1.0, rng.normal(30, 6)))

            utilization = float(np.clip(rng.normal(65, 20), 5, 100))
            queue_depth = float(max(1.0, rng.normal(profile["queue_depth"], 2.0)))

            host_writes = float(max(1.0, rng.normal(profile["host_writes_mean"], 50)))
            nand_writes = float(max(1.0, rng.normal(profile["host_writes_mean"] * 1.5, 80)))

            thermal_events = int(max(0, rng.poisson(profile["thermal_lambda"])))

            out.append(
                {
                    "run_id": run["run_id"],
                    "event_ts": (start + pd.Timedelta(minutes=5 * i)).isoformat(),
                    "device_id": f"{run['device_group']}_{rng.integers(1000,9999)}",
                    "device_name": device_name,
                    "storage_tier": storage_tier,
                    "soc_model": soc_model,
                    "read_iops": read_iops,
                    "write_iops": write_iops,
                    "read_throughput_mb_s": read_tp,
                    "write_throughput_mb_s": write_tp,
                    "latency_read_p99_ms": latency_read,
                    "latency_write_p99_ms": latency_write,
                    "utilization_pct": utilization,
                    "nvme_queue_depth": queue_depth,
                    "host_writes_mb": host_writes,
                    "nand_writes_mb": nand_writes,
                    "thermal_throttling_events": thermal_events,
                }
            )

    return pd.DataFrame(out)

def main():
    parser = argparse.ArgumentParser(description="Generate raw Apple SSD telemetry data")
    parser.add_argument("--rows-per-group", type=int, default=100)
    parser.add_argument("--runs-per-group", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("data/generated"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    runs_df = _generate_run_rows(rng, args.runs_per_group)
    metrics_df = _generate_metric_rows(rng, runs_df, args.rows_per_group)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs_df.to_csv(args.output_dir / "runs.csv", index=False)
    metrics_df.to_csv(args.output_dir / "raw_device_metrics.csv", index=False)

    print(f"Generated {len(metrics_df)} rows")


if __name__ == "__main__":
    main()