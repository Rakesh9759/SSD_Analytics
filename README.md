# SSD Analytics (Synthetic + Corrupted Telemetry)

This project generates a synthetic SSD telemetry dataset and injects realistic data quality issues for profiling, validation, and EDA practice.

## Script to use

Use only:
- `generate_corrupted_telemetry.py`

Run example:

```bash
python generate_corrupted_telemetry.py --rows-per-group 100 --runs-per-group 2 --seed 42
```

Default output:
- `data/generated/corrupted_dataset.csv`

## Dataset fields

- `run_id`: UUID for a benchmark/test run.
- `event_ts`: Event timestamp in ISO 8601 UTC format.
- `device_id`: Synthetic per-device identifier.
- `device_name`: Device model name (for example iPhone/Mac variants).
- `storage_tier`: Nominal storage capacity tier (for example 128GB, 256GB).
- `soc_model`: SoC/chip model label (A-series/M-series style).
- `read_iops`: Random read operations per second.
- `write_iops`: Random write operations per second.
- `read_throughput_mb_s`: Read throughput in MB/s.
- `write_throughput_mb_s`: Write throughput in MB/s.
- `latency_read_p99_ms`: p99 read latency in milliseconds.
- `latency_write_p99_ms`: p99 write latency in milliseconds.
- `utilization_pct`: Device utilization percentage.
- `nvme_queue_depth`: NVMe queue depth.
- `cpu_usage_pct`: CPU usage percentage.
- `memory_usage_pct`: Memory usage percentage.
- `io_wait_pct`: I/O wait percentage.
- `host_writes_mb`: Host write volume in MB.
- `nand_writes_mb`: NAND write volume in MB.
- `thermal_throttling_events`: Count of thermal throttling events.

## Injected anomaly types

- Missing values
- Wrong data types in numeric columns
- Out-of-range numeric values
- Invalid categorical values
- Corrupted timestamps
- Duplicate rows
- Row shuffling

## Helpful references

- ISO 8601 datetime format: https://www.iso.org/iso-8601-date-and-time-format.html
- NVMe concepts (queue depth, latency, throughput): https://nvmexpress.org/specifications/


The orchestration layer is implemented in `src/pipeline/dag.py` as an
Airflow-style DAG skeleton with explicit stage functions.

Stages:
- `generate_data`
- `corrupt_data`
- `validate_data`
- `clean_data`
- `feature_engineering`
- `anomaly_detection`
- `root_cause`

Run example:

```bash
python -m src.pipeline.dag --rows-per-group 100 --runs-per-group 2 --seed 42
```

Generated artifacts (under `data/generated/`):
- `pipeline_raw_dataset.csv`
- `pipeline_corrupted_dataset.csv`
- `pipeline_cleaned_dataset.csv`
- `pipeline_root_cause_output.csv`
- `pipeline_dq_score_table.csv`
- `pipeline_stage_metrics.csv`

Architecture diagram (ASCII):

```text
				 +------------------------+
				 | generate_data          |
				 | (raw telemetry)        |
				 +-----------+------------+
							 |
							 v
				 +-----------+------------+
				 | corrupt_data           |
				 | + anomaly injection    |
				 +-----------+------------+
							 |
							 v
				 +-----------+------------+
				 | validate_data          |
				 | (DQ checks + scoring)  |
				 +-----------+------------+
							 |
							 v
				 +-----------+------------+
				 | clean_data             |
				 | (coerce + dedup + clip)|
				 +-----------+------------+
							 |
							 v
				 +-----------+------------+
				 | feature_engineering    |
				 | (derived telemetry)    |
				 +-----------+------------+
							 |
							 v
				 +-----------+------------+
				 | anomaly_detection      |
				 | (IF + multivariate z)  |
				 +-----------+------------+
							 |
							 v
				 +-----------+------------+
				 | root_cause             |
				 | (label + confidence)   |
				 +-----------+------------+
							 |
							 v
				 +-----------+------------+
				 | CSV artifacts          |
				 | in data/generated/     |
				 +------------------------+
```
