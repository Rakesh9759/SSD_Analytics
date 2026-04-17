"""Anomaly detectors for SSD telemetry: Isolation Forest + multivariate z-score."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


# Default feature columns
DEFAULT_FEATURES = [
    "latency_write_p99_ms",
    "latency_read_p99_ms",
    "cpu_usage_pct",
    "io_wait_pct",
    "memory_usage_pct",
    "nvme_queue_depth",
    "total_iops",
    "write_amplification",
    "saturation_score",
    "io_mix",
    "cpu_to_io_ratio",
    "io_wait_to_latency_ratio",
    "burstiness",
]


class IsolationForestDetector:
    """sklearn IsolationForest with a pandas-friendly interface."""

    def __init__(
        self,
        features: list[str] | None = None,
        contamination: float = 0.1,
        random_state: int = 42,
        **if_kwargs,
    ):
        self.features = features or list(DEFAULT_FEATURES)
        self.contamination = contamination
        self.random_state = random_state
        self._model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            **if_kwargs,
        )
        self._fitted = False

    def _get_X(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        cols = [c for c in self.features if c in df.columns]
        X = df[cols].copy()
        X = X.fillna(X.median(numeric_only=True))  # impute NaN with median
        return X, cols

    def fit(self, df: pd.DataFrame) -> "IsolationForestDetector":
        X, _ = self._get_X(df)
        self._model.fit(X)
        self._fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")
        X, _ = self._get_X(df)
        out = df.copy()
        out["anomaly_score"] = -self._model.score_samples(X)  # higher = more anomalous
        out["if_anomaly"] = self._model.predict(X) == -1  # IF returns -1 for anomaly
        return out

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).predict(df)


class MultivariateZScoreDetector:
    """Flags rows where any feature's z-score exceeds a threshold."""

    def __init__(
        self,
        features: list[str] | None = None,
        threshold: float = 3.0,
    ):
        self.features = features or list(DEFAULT_FEATURES)
        self.threshold = threshold
        self._means: pd.Series | None = None
        self._stds: pd.Series | None = None

    def _get_cols(self, df: pd.DataFrame) -> list[str]:
        return [c for c in self.features if c in df.columns]

    def fit(self, df: pd.DataFrame) -> "MultivariateZScoreDetector":
        cols = self._get_cols(df)
        self._means = df[cols].mean()
        self._stds = df[cols].std().replace(0, np.nan)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._means is None:
            raise RuntimeError("Call fit() before predict().")
        cols = self._get_cols(df)
        out = df.copy()
        z = (df[cols] - self._means[cols]) / self._stds[cols]
        out["mz_max_zscore"] = z.abs().max(axis=1)
        out["mz_anomaly"] = out["mz_max_zscore"] > self.threshold
        return out

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).predict(df)


def run_all_detectors(
    df: pd.DataFrame,
    contamination: float = 0.1,
    zscore_threshold: float = 3.0,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """Fit both detectors and add anomaly_score, if_anomaly, mz_max_zscore, mz_anomaly, detected_anomaly."""
    ifd = IsolationForestDetector(
        features=features, contamination=contamination
    )
    mzd = MultivariateZScoreDetector(
        features=features, threshold=zscore_threshold
    )

    out = ifd.fit_predict(df)
    out = mzd.fit_predict(out)
    out["detected_anomaly"] = out["if_anomaly"] | out["mz_anomaly"]
    return out


def get_detection_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return flagged-count comparison table. Requires run_all_detectors() first."""
    required = {"if_anomaly", "mz_anomaly", "detected_anomaly"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns — call run_all_detectors() first: {missing}")

    total = len(df)
    rows = [
        {
            "detector": "isolation_forest",
            "flagged": int(df["if_anomaly"].sum()),
            "pct": round(df["if_anomaly"].mean() * 100, 2),
        },
        {
            "detector": "multivariate_zscore",
            "flagged": int(df["mz_anomaly"].sum()),
            "pct": round(df["mz_anomaly"].mean() * 100, 2),
        },
        {
            "detector": "combined (union)",
            "flagged": int(df["detected_anomaly"].sum()),
            "pct": round(df["detected_anomaly"].mean() * 100, 2),
        },
    ]
    return pd.DataFrame(rows)
