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

NUMERIC_COLUMNS = [
    "read_iops","write_iops","read_throughput_mb_s","write_throughput_mb_s",
    "latency_read_p99_ms","latency_write_p99_ms","utilization_pct",
    "nvme_queue_depth","host_writes_mb","nand_writes_mb",
    "thermal_throttling_events", "cpu_usage_pct","memory_usage_pct","io_wait_pct"
]

# Fraction of total rows to replace with each anomaly scenario
ANOMALY_FRACTIONS = {
    "disk_bottleneck": 0.06,     # latency high + io_wait high
    "cpu_bottleneck": 0.06,      # latency high + cpu high
    "memory_pressure": 0.05,     # latency high + memory high
    "suspicious_unknown": 0.04,  # latency high, all other signals normal
}

WRONG_TYPE_TOKENS = ["error","N/A","bad_data","null",""]

INVALID_DEVICE_NAMES = ["iphone??","unknown","", "123"]
INVALID_STORAGE = ["999GB","64TB","unknown",""]
INVALID_SOC = ["M9","A99","chip_x",""]

# =========================
# DATA GENERATION
# =========================

def generate_data(rng, runs_per_group, rows_per_group):

    rows = []
    now = datetime.now(timezone.utc)

    for group in GROUP_PROFILES:
        for r in range(runs_per_group):

            run_id = str(uuid4())
            start = now - timedelta(days=r*2)

            for i in range(rows_per_group):

                profile = GROUP_PROFILES[group]

                rows.append({
                    "run_id": run_id,
                    "event_ts": (start + timedelta(minutes=5*i)).isoformat(),

                    "device_id": f"{group}_{rng.integers(1000,9999)}",
                    "device_name": rng.choice(APPLE_DEVICE_MAP[group]),
                    "storage_tier": rng.choice(STORAGE_TIERS),
                    "soc_model": rng.choice(SOC_MODELS),

                    # I/O
                    "read_iops": float(max(10, rng.normal(800,200))),
                    "write_iops": float(max(10, rng.normal(600,150))),
                    "read_throughput_mb_s": float(max(1, rng.normal(120,30))),
                    "write_throughput_mb_s": float(max(1, rng.normal(90,25))),

                    # Latency
                    "latency_read_p99_ms": float(max(1, rng.normal(20,5))),
                    "latency_write_p99_ms": float(max(1, rng.normal(30,6))),

                    # Disk/System pressure
                    "utilization_pct": float(np.clip(rng.normal(65,20),5,100)),
                    "nvme_queue_depth": float(max(1, rng.normal(profile["queue_depth"],2))),

                    # NEW system metrics
                    "cpu_usage_pct": float(np.clip(rng.normal(60,20),5,100)),
                    "memory_usage_pct": float(np.clip(rng.normal(70,15),10,100)),
                    "io_wait_pct": float(np.clip(rng.normal(10,5),0,50)),

                    # Endurance
                    "host_writes_mb": float(max(1, rng.normal(profile["host_writes_mean"],50))),
                    "nand_writes_mb": float(max(1, rng.normal(profile["host_writes_mean"]*1.5,80))),

                    # Events
                    "thermal_throttling_events": int(max(0, rng.poisson(profile["thermal_lambda"]))),
                })

    return pd.DataFrame(rows)


# =========================
# ANOMALY INJECTION
# =========================

def inject_anomaly_scenarios(df: pd.DataFrame, rng) -> pd.DataFrame:
    """
    Overwrite a fraction of rows with realistic anomaly scenarios so that the
    root cause engine has meaningful signal to classify.

    Each scenario sets latency_write_p99_ms well above the 80 ms threshold and
    then raises the co-occurring signal that defines the root cause label:

        disk_bottleneck    — latency > 80 ms  +  io_wait > 20 %
        cpu_bottleneck     — latency > 80 ms  +  cpu > 85 %
        memory_pressure    — latency > 80 ms  +  memory > 85 %
        suspicious_unknown — latency > 80 ms  +  all others normal

    The rows are chosen randomly so scenarios are spread across device groups
    and timestamps, making the dataset representative rather than clustered.
    """
    out = df.copy()
    n = len(out)
    available = set(out.index.tolist())

    def sample_idx(frac):
        size = max(1, int(n * frac))
        chosen = rng.choice(sorted(available), size=min(size, len(available)), replace=False)
        available.difference_update(chosen)
        return chosen

    # --- disk_bottleneck ---
    idx = sample_idx(ANOMALY_FRACTIONS["disk_bottleneck"])
    out.loc[idx, "latency_write_p99_ms"] = rng.uniform(85, 180, len(idx))
    out.loc[idx, "latency_read_p99_ms"]  = rng.uniform(60, 140, len(idx))
    out.loc[idx, "io_wait_pct"]          = rng.uniform(22, 55,  len(idx))
    out.loc[idx, "cpu_usage_pct"]        = rng.uniform(30, 70,  len(idx))
    out.loc[idx, "memory_usage_pct"]     = rng.uniform(40, 80,  len(idx))
    out.loc[idx, "nvme_queue_depth"]     = rng.uniform(25, 64,  len(idx))

    # --- cpu_bottleneck ---
    idx = sample_idx(ANOMALY_FRACTIONS["cpu_bottleneck"])
    out.loc[idx, "latency_write_p99_ms"] = rng.uniform(85, 160, len(idx))
    out.loc[idx, "latency_read_p99_ms"]  = rng.uniform(60, 120, len(idx))
    out.loc[idx, "cpu_usage_pct"]        = rng.uniform(88, 100, len(idx))
    out.loc[idx, "io_wait_pct"]          = rng.uniform(2,  15,  len(idx))
    out.loc[idx, "memory_usage_pct"]     = rng.uniform(40, 80,  len(idx))

    # --- memory_pressure ---
    idx = sample_idx(ANOMALY_FRACTIONS["memory_pressure"])
    out.loc[idx, "latency_write_p99_ms"] = rng.uniform(85, 150, len(idx))
    out.loc[idx, "latency_read_p99_ms"]  = rng.uniform(60, 110, len(idx))
    out.loc[idx, "memory_usage_pct"]     = rng.uniform(88, 100, len(idx))
    out.loc[idx, "cpu_usage_pct"]        = rng.uniform(30, 70,  len(idx))
    out.loc[idx, "io_wait_pct"]          = rng.uniform(2,  15,  len(idx))

    # --- suspicious_unknown ---
    idx = sample_idx(ANOMALY_FRACTIONS["suspicious_unknown"])
    out.loc[idx, "latency_write_p99_ms"] = rng.uniform(85, 200, len(idx))
    out.loc[idx, "latency_read_p99_ms"]  = rng.uniform(60, 160, len(idx))
    out.loc[idx, "cpu_usage_pct"]        = rng.uniform(10, 40,  len(idx))
    out.loc[idx, "io_wait_pct"]          = rng.uniform(0,  8,   len(idx))
    out.loc[idx, "memory_usage_pct"]     = rng.uniform(20, 60,  len(idx))

    return out


# =========================
# DATA CORRUPTION
# =========================

def corrupt_data(df, rng):

    out = df.copy()
    n = len(out)

    def pick(frac):
        return rng.choice(out.index, size=max(1,int(n*frac)), replace=False)

    # Missing
    for col in out.columns:
        out.loc[pick(0.06), col] = np.nan

    # Wrong types
    for col in NUMERIC_COLUMNS:
        if col in out.columns:
            out[col] = out[col].astype(object)
            out.loc[pick(0.03), col] = rng.choice(WRONG_TYPE_TOKENS)

    # Out-of-range
    out.loc[pick(0.02), "utilization_pct"] = 150
    out.loc[pick(0.02), "latency_read_p99_ms"] = -10
    out.loc[pick(0.02), "read_iops"] = 9999999
    out.loc[pick(0.02), "cpu_usage_pct"] = 200
    out.loc[pick(0.02), "io_wait_pct"] = -5

    # Invalid categories
    out.loc[pick(0.03), "device_name"] = rng.choice(INVALID_DEVICE_NAMES)
    out.loc[pick(0.03), "storage_tier"] = rng.choice(INVALID_STORAGE)
    out.loc[pick(0.03), "soc_model"] = rng.choice(INVALID_SOC)

    # Timestamp issues
    out.loc[pick(0.02), "event_ts"] = "not_a_timestamp"

    # Duplicates
    dup = out.sample(frac=0.03, random_state=42)
    out = pd.concat([out, dup], ignore_index=True)

    # Shuffle
    out = out.sample(frac=1).reset_index(drop=True)

    return out


# =========================
# MAIN FUNCTION
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows-per-group", type=int, default=100)
    parser.add_argument("--runs-per-group", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("data/generated/corrupted_dataset.csv"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    df = generate_data(rng, args.runs_per_group, args.rows_per_group)
    df = inject_anomaly_scenarios(df, rng)
    df = corrupt_data(df, rng)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"Final dataset rows: {len(df)}")


if __name__ == "__main__":
    main()