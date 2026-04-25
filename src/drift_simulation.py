"""IEEE CIS - Drift Simulation.

Three injectable drift modes:
  1. ``time_split``       - split train/test along TransactionDT.
  2. ``new_pattern``      - synthesise a new fraud cluster absent in train.
  3. ``feature_shift``    - perturb feature distributions and log
                            importance shift between models trained on each
                            slice.

Outputs PSI / KS diagnostics, drifted-feature ranking, and a feature
importance shift JSON to ``outputs/drift_report.json``.

Usage::

    python -m src.drift_simulation --input train_transaction.csv \
        --inject-new-pattern --injection-fraction 0.05
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy import stats

LOG = logging.getLogger("drift")

LABEL = "isFraud"
ID_COLS = ["TransactionID"]


@dataclass
class DriftReport:
    psi: Dict[str, float]
    ks: Dict[str, Dict[str, float]]
    drifted_features: List[str]
    summary: Dict[str, float]


def population_stability_index(
    expected: np.ndarray, actual: np.ndarray, bins: int = 10,
) -> float:
    """PSI between two 1-D distributions.

    Uses quantile-based bin edges from the *expected* (reference) sample
    to keep bin populations roughly equal in the baseline.
    """
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if expected.size == 0 or actual.size == 0:
        return 0.0
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if edges.size < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e_hist, _ = np.histogram(expected, bins=edges)
    a_hist, _ = np.histogram(actual, bins=edges)
    e_pct = np.clip(e_hist / max(e_hist.sum(), 1), 1e-6, None)
    a_pct = np.clip(a_hist / max(a_hist.sum(), 1), 1e-6, None)
    return float(((a_pct - e_pct) * np.log(a_pct / e_pct)).sum())


def detect_drift(
    reference: pd.DataFrame, current: pd.DataFrame, threshold: float = 0.20,
) -> DriftReport:
    """Compute PSI + KS for every shared numeric column."""
    num_cols = reference.select_dtypes(include=[np.number]).columns
    num_cols = [c for c in num_cols if c not in (LABEL, *ID_COLS)]
    psi_map: Dict[str, float] = {}
    ks_map: Dict[str, Dict[str, float]] = {}

    for c in num_cols:
        if c not in current.columns:
            continue
        ref_vals = reference[c].astype(float).values
        cur_vals = current[c].astype(float).values
        psi_map[c] = population_stability_index(ref_vals, cur_vals)
        if ref_vals.size > 30 and cur_vals.size > 30:
            r = ref_vals[~np.isnan(ref_vals)]
            a = cur_vals[~np.isnan(cur_vals)]
            if r.size and a.size:
                ks = stats.ks_2samp(r, a)
                ks_map[c] = {
                    "statistic": float(ks.statistic),
                    "pvalue": float(ks.pvalue),
                }

    drifted = sorted(
        [f for f, v in psi_map.items() if v > threshold],
        key=lambda f: -psi_map[f],
    )
    summary = {
        "psi_max": max(psi_map.values()) if psi_map else 0.0,
        "psi_mean": float(np.mean(list(psi_map.values()))) if psi_map else 0.0,
        "n_drifted_features": len(drifted),
        "n_features_evaluated": len(psi_map),
    }
    return DriftReport(psi=psi_map, ks=ks_map, drifted_features=drifted, summary=summary)


def time_split_drift(
    df: pd.DataFrame, split_quantile: float = 0.7,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Time-based split using TransactionDT."""
    if "TransactionDT" not in df.columns:
        raise KeyError("TransactionDT required for time-based drift simulation")
    cut = df["TransactionDT"].quantile(split_quantile)
    early = df[df["TransactionDT"] <= cut].copy()
    late = df[df["TransactionDT"] > cut].copy()
    LOG.info(
        "Time-split: early=%d (fraud_rate=%.4f) | late=%d (fraud_rate=%.4f)",
        len(early), early[LABEL].mean(),
        len(late), late[LABEL].mean(),
    )
    return early, late


def inject_new_fraud_pattern(
    df: pd.DataFrame, fraction: float = 0.05, seed: int = 7,
) -> pd.DataFrame:
    """Synthesise a brand-new fraud cluster in TransactionAmt + V-block space."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    n_new = int(len(out) * fraction)
    if n_new == 0:
        return out
    idx = rng.choice(out.index, size=n_new, replace=False)
    out.loc[idx, LABEL] = 1
    if "TransactionAmt" in out.columns:
        out.loc[idx, "TransactionAmt"] = rng.uniform(950, 1050, size=n_new)
    v_cols = [c for c in out.columns if c.startswith("V") and c[1:].isdigit()][:20]
    if v_cols:
        shift = rng.normal(loc=2.5, scale=0.5, size=(n_new, len(v_cols)))
        out.loc[idx, v_cols] = out.loc[idx, v_cols].fillna(0).values + shift
    LOG.info("Injected %d synthetic fraud rows (new pattern, seed=%d)", n_new, seed)
    return out


def feature_importance_shift(
    reference: pd.DataFrame, current: pd.DataFrame,
) -> Dict[str, Dict[str, float]]:
    """Refit LightGBM on each slice; log normalised gain delta per feature."""

    def _fit(df: pd.DataFrame) -> Tuple[LGBMClassifier, List[str]]:
        y = df[LABEL].astype(int)
        X = df.drop(columns=[LABEL] + [c for c in ID_COLS if c in df.columns])
        X = X.select_dtypes(include=[np.number]).fillna(0)
        clf = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=64,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        clf.fit(X, y)
        return clf, list(X.columns)

    LOG.info("Fitting reference LightGBM for importance baseline")
    ref_clf, ref_feats = _fit(reference)
    LOG.info("Fitting current LightGBM for importance comparison")
    cur_clf, cur_feats = _fit(current)
    common = [f for f in ref_feats if f in cur_feats]
    ref_imp = dict(
        zip(ref_feats, ref_clf.booster_.feature_importance(importance_type="gain"))
    )
    cur_imp = dict(
        zip(cur_feats, cur_clf.booster_.feature_importance(importance_type="gain"))
    )
    ref_total = sum(ref_imp.values()) or 1.0
    cur_total = sum(cur_imp.values()) or 1.0
    shift = {}
    for f in common:
        rp = ref_imp[f] / ref_total
        cp = cur_imp[f] / cur_total
        shift[f] = {
            "reference_share": float(rp),
            "current_share": float(cp),
            "delta": float(cp - rp),
        }
    ranked = sorted(shift.items(), key=lambda kv: -abs(kv[1]["delta"]))[:25]
    return dict(ranked)


def run(args: argparse.Namespace) -> Dict:
    LOG.info("Loading %s", args.input)
    df = (
        pd.read_csv(args.input)
        if args.input.endswith(".csv")
        else pd.read_parquet(args.input)
    )

    early, late = time_split_drift(df, split_quantile=args.split_quantile)
    if args.inject_new_pattern:
        late = inject_new_fraud_pattern(late, fraction=args.injection_fraction)

    drift = detect_drift(early, late, threshold=args.psi_threshold)
    LOG.info("PSI summary: %s", drift.summary)
    LOG.info("Top drifted features: %s", drift.drifted_features[:10])

    shift = feature_importance_shift(early, late)
    LOG.info("Top importance-shift features: %s", list(shift.keys())[:10])

    os.makedirs("outputs", exist_ok=True)
    payload = {
        "drift_report": asdict(drift),
        "feature_importance_shift": shift,
    }
    with open("outputs/drift_report.json", "w") as f:
        json.dump(payload, f, indent=2)

    # Drop a slim live_metrics.json so the Prometheus exporter can scrape drift.
    live = {
        "feature_psi_max": drift.summary["psi_max"],
        "prediction_psi": drift.summary["psi_mean"],
        "feature_ks": {f: v["statistic"] for f, v in list(drift.ks.items())[:50]},
    }
    with open("outputs/live_metrics_drift.json", "w") as f:
        json.dump(live, f, indent=2)
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Drift simulation harness.")
    p.add_argument("--input", required=True)
    p.add_argument("--split-quantile", type=float, default=0.7)
    p.add_argument(
        "--inject-new-pattern", action="store_true", default=True,
        help="Synthesise a new fraud cluster in the late slice.",
    )
    p.add_argument("--injection-fraction", type=float, default=0.05)
    p.add_argument("--psi-threshold", type=float, default=0.20)
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(parse_args())
