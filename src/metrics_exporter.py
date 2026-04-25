"""Prometheus metrics exporter for the IEEE CIS fraud-detection service.

Run as a sidecar or standalone process::

    python -m src.metrics_exporter --port 9000 --metrics-file outputs/live_metrics.json

The inference server imports the request-side instruments
(``REQUEST_LATENCY``, ``REQUESTS_TOTAL``, ``CONFIDENCE``) directly from
this module so a single registry covers both training and serving.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

from prometheus_client import Counter, Gauge, Histogram, start_http_server

LOG = logging.getLogger("metrics_exporter")

# Model-performance gauges - refreshed by training/evaluation jobs.
RECALL = Gauge("fraud_recall", "Current model recall on the evaluation slice")
PRECISION = Gauge("fraud_precision", "Current model precision")
F1 = Gauge("fraud_f1", "Current model F1")
AUC = Gauge("fraud_auc_roc", "Current model AUC-ROC")
FPR = Gauge("fraud_false_positive_rate", "False-positive rate")

# Drift gauges - refreshed by drift_simulation / drift-monitor jobs.
FEATURE_PSI_MAX = Gauge(
    "feature_psi_max", "Maximum per-feature PSI vs the training distribution",
)
PREDICTION_PSI = Gauge(
    "prediction_psi", "PSI of the prediction-score distribution vs baseline",
)
FEATURE_KS = Gauge(
    "feature_ks_statistic",
    "Per-feature Kolmogorov-Smirnov statistic vs training",
    ["feature"],
)

# Live request instruments - shared with the inference server.
REQUEST_LATENCY = Histogram(
    "inference_request_duration_seconds",
    "Inference request duration (seconds)",
    buckets=(0.005, 0.010, 0.025, 0.050, 0.100, 0.250, 0.500, 1.0, 2.5, 5.0),
)
REQUESTS_TOTAL = Counter(
    "inference_requests_total",
    "Inference request count, labelled by status",
    ["status"],
)
CONFIDENCE = Histogram(
    "fraud_confidence",
    "Distribution of fraud probability scores returned by the model",
    buckets=(0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
)


def refresh_from_file(path: Path) -> None:
    """Read JSON metrics dropped by training/evaluation/drift jobs."""
    if not path.exists():
        return
    try:
        with path.open() as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        LOG.warning("Unparseable metrics file at %s: %s", path, e)
        return

    gauge_map = {
        "recall": RECALL,
        "precision": PRECISION,
        "f1": F1,
        "auc_roc": AUC,
        "false_positive_rate": FPR,
        "feature_psi_max": FEATURE_PSI_MAX,
        "prediction_psi": PREDICTION_PSI,
    }
    for key, gauge in gauge_map.items():
        if key in data:
            try:
                gauge.set(float(data[key]))
            except (TypeError, ValueError):
                LOG.warning("Bad value for %s: %r", key, data[key])

    for feat, ks in data.get("feature_ks", {}).items():
        try:
            FEATURE_KS.labels(feature=feat).set(float(ks))
        except (TypeError, ValueError):
            continue


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prometheus metrics exporter.")
    p.add_argument(
        "--port", type=int, default=int(os.environ.get("METRICS_PORT", 9000)),
        help="HTTP port for Prometheus scrape endpoint.",
    )
    p.add_argument(
        "--metrics-file", default="outputs/live_metrics.json",
        help="Path to JSON metrics file that training/eval jobs refresh.",
    )
    p.add_argument(
        "--interval", type=float, default=15.0,
        help="Refresh interval in seconds.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    start_http_server(args.port)
    LOG.info("Prometheus exporter listening on :%d", args.port)
    LOG.info("Polling metrics file: %s every %ss", args.metrics_file, args.interval)
    path = Path(args.metrics_file)
    while True:
        refresh_from_file(path)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
