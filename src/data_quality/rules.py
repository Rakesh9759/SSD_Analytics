"""Importable constants for SSD telemetry data quality checks."""

EXPECTED_COLS = [
    'run_id', 'event_ts', 'device_id', 'device_name', 'storage_tier', 'soc_model',
    'read_iops', 'write_iops', 'read_throughput_mb_s', 'write_throughput_mb_s',
    'latency_read_p99_ms', 'latency_write_p99_ms', 'utilization_pct', 'nvme_queue_depth',
    'cpu_usage_pct', 'memory_usage_pct', 'io_wait_pct', 'host_writes_mb',
    'nand_writes_mb', 'thermal_throttling_events',
]

NUMERIC_COLS = [
    'read_iops', 'write_iops', 'read_throughput_mb_s', 'write_throughput_mb_s',
    'latency_read_p99_ms', 'latency_write_p99_ms', 'utilization_pct', 'nvme_queue_depth',
    'cpu_usage_pct', 'memory_usage_pct', 'io_wait_pct', 'host_writes_mb',
    'nand_writes_mb', 'thermal_throttling_events',
]

VALID_STORAGE = {'128GB', '256GB', '512GB', '1TB'}

VALID_SOC = {'A15', 'A16', 'A17', 'M1', 'M2'}

RANGE_RULES = {
    'cpu_usage_pct':        (0, 100),
    'memory_usage_pct':     (0, 100),
    'utilization_pct':      (0, 100),
    'io_wait_pct':          (0, 100),
    'latency_read_p99_ms':  (0, None),
    'latency_write_p99_ms': (0, None),
    'read_iops':            (0, None),
    'write_iops':           (0, None),
    'nvme_queue_depth':     (0, None),
}
