"""IEEE CIS - Retraining Strategy.

Compares three retraining policies on a chronological replay of the
data and prints a stability / cost / performance comparison table:

  - **periodic**   - fixed cadence (e.g. every N windows)
  - **threshold**  - retrain only when recall drops below T or PSI > T_d
  - **hybrid**     - threshold-driven, with a periodic safety floor

Outputs ``outputs/retraining_comparison.json`` with per-window metrics
and total cost decomposition.

Usage::

    python -m src.retraining_strategy --input outputs/features.parquet --windows 8
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
)

LOG = logging.getLogger("retraining")

LABEL = "isFraud"
ID_COLS = ["TransactionID"]

# Cost model (USD).
RETRAIN_COST_USD = 250.0
PERFORMANCE_LOSS_PER_RECALL_POINT = 1500.0  # cost per 0.01 recall lost vs SLO
RECALL_SLO = 0.85


def population_stability_index(
    expected: np.ndarray, actual: np.ndarray, bins: int = 10,
) -> float:
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


@dataclass
class WindowResult:
    window: int
    retrained: bool
    reason: str
    recall: float
    precision: float
    f1: float
    auc_roc: float
    psi_max: float


@dataclass
class StrategyOutcome:
    name: str
    retrains: int
    avg_recall: float
    min_recall: float
    avg_auc: float
    recall_std: float
    total_retrain_cost: float
    total_performance_cost: float
    total_cost: float
    windows: List[WindowResult]


def chronological_windows(df: pd.DataFrame, n_windows: int = 8) -> List[pd.DataFrame]:
    if "TransactionDT" in df.columns:
        df = df.sort_values("TransactionDT").reset_index(drop=True)
    return [w.reset_index(drop=True) for w in np.array_split(df, n_windows)]


def fit_model(train_df: pd.DataFrame) -> LGBMClassifier:
    y = train_df[LABEL].astype(int)
    X = train_df.drop(columns=[LABEL] + [c for c in ID_COLS if c in train_df.columns])
    X = X.select_dtypes(include=[np.number]).fillna(0)
    clf = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=64,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    clf.fit(X, y)
    clf.fraud_features_ = list(X.columns)  # type: ignore[attr-defined]
    return clf


def score(model: LGBMClassifier, eval_df: pd.DataFrame) -> Dict[str, float]:
    feats: List[str] = model.fraud_features_  # type: ignore[attr-defined]
    X = eval_df.reindex(columns=feats).fillna(0)
    y = eval_df[LABEL].astype(int)
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    return {
        "recall": float(recall_score(y, preds, zero_division=0)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "auc": float(roc_auc_score(y, probs)) if len(set(y)) > 1 else float("nan"),
    }


def feature_psi(reference: pd.DataFrame, current: pd.DataFrame) -> float:
    nums = reference.select_dtypes(include=[np.number]).columns.tolist()
    nums = [c for c in nums if c not in (LABEL, *ID_COLS)]
    psi_vals = []
    for c in nums[:50]:
        if c in current.columns:
            psi_vals.append(
                population_stability_index(
                    reference[c].astype(float).values,
                    current[c].astype(float).values,
                )
            )
    return float(max(psi_vals)) if psi_vals else 0.0


def simulate(
    name: str,
    windows: List[pd.DataFrame],
    decide: Callable[[int, Dict[str, float], float], Tuple[bool, str]],
) -> StrategyOutcome:
    LOG.info("Strategy: %s", name)
    train_buf = pd.concat(windows[:2]).reset_index(drop=True)
    model = fit_model(train_buf)
    retrains = 0
    history: List[WindowResult] = []

    for i, w in enumerate(windows[2:], start=2):
        psi = feature_psi(train_buf, w)
        metrics = score(model, w)
        retrain, reason = decide(i, metrics, psi)
        if retrain:
            retrains += 1
            train_buf = pd.concat([train_buf, w]).tail(60_000).reset_index(drop=True)
            model = fit_model(train_buf)
        history.append(
            WindowResult(
                window=i,
                retrained=retrain,
                reason=reason,
                recall=metrics["recall"],
                precision=metrics["precision"],
                f1=metrics["f1"],
                auc_roc=metrics["auc"],
                psi_max=psi,
            )
        )

    recalls = np.array([h.recall for h in history])
    avg_recall = float(recalls.mean()) if recalls.size else 0.0
    perf_loss_points = max(0.0, RECALL_SLO - avg_recall) * 100
    perf_cost = perf_loss_points * PERFORMANCE_LOSS_PER_RECALL_POINT
    retrain_cost = retrains * RETRAIN_COST_USD
    return StrategyOutcome(
        name=name,
        retrains=retrains,
        avg_recall=avg_recall,
        min_recall=float(recalls.min()) if recalls.size else 0.0,
        avg_auc=float(np.nanmean([h.auc_roc for h in history])) if history else 0.0,
        recall_std=float(recalls.std()) if recalls.size else 0.0,
        total_retrain_cost=retrain_cost,
        total_performance_cost=perf_cost,
        total_cost=retrain_cost + perf_cost,
        windows=history,
    )


def policy_periodic(cadence: int = 3) -> Callable:
    state = {"last": 1}

    def _p(i: int, m: Dict[str, float], psi: float) -> Tuple[bool, str]:
        if i - state["last"] >= cadence:
            state["last"] = i
            return True, "periodic_cadence"
        return False, "skip"
    return _p


def policy_threshold(
    recall_thr: float = 0.78, psi_thr: float = 0.25,
) -> Callable:
    def _p(i: int, m: Dict[str, float], psi: float) -> Tuple[bool, str]:
        if m["recall"] < recall_thr:
            return True, f"recall<{recall_thr}({m['recall']:.3f})"
        if psi > psi_thr:
            return True, f"psi>{psi_thr}({psi:.3f})"
        return False, "stable"
    return _p


def policy_hybrid(
    cadence: int = 4, recall_thr: float = 0.78, psi_thr: float = 0.25,
) -> Callable:
    p_thr = policy_threshold(recall_thr, psi_thr)
    p_per = policy_periodic(cadence)

    def _p(i: int, m: Dict[str, float], psi: float) -> Tuple[bool, str]:
        retr, reason = p_thr(i, m, psi)
        if retr:
            return retr, f"hybrid:{reason}"
        retr, reason = p_per(i, m, psi)
        return retr, f"hybrid:{reason}"
    return _p


def render_table(outcomes: List[StrategyOutcome]) -> str:
    header = (
        "strategy        retrains  avg_recall  min_recall  recall_std  "
        "retrain_$  perf_$    total_$"
    )
    lines = [header, "-" * len(header)]
    for o in outcomes:
        lines.append(
            f"{o.name:<14} {o.retrains:>8}  {o.avg_recall:>10.4f}  "
            f"{o.min_recall:>10.4f}  {o.recall_std:>10.4f}  "
            f"${o.total_retrain_cost:>8,.0f}  ${o.total_performance_cost:>7,.0f}  "
            f"${o.total_cost:>8,.0f}"
        )
    cheapest = min(outcomes, key=lambda o: o.total_cost)
    lines.append(
        f"\nLowest-cost policy: {cheapest.name} (${cheapest.total_cost:,.0f})"
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> Dict:
    LOG.info("Loading %s", args.input)
    df = (
        pd.read_csv(args.input)
        if args.input.endswith(".csv")
        else pd.read_parquet(args.input)
    )
    windows = chronological_windows(df, n_windows=args.windows)
    LOG.info("Built %d chronological windows of ~%d rows", len(windows), len(windows[0]))

    outcomes = [
        simulate("periodic", windows, policy_periodic(cadence=args.cadence)),
        simulate("threshold", windows, policy_threshold()),
        simulate("hybrid", windows, policy_hybrid(cadence=args.cadence)),
    ]
    table = render_table(outcomes)
    print(table)

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/retraining_comparison.json", "w") as f:
        json.dump(
            {"outcomes": [asdict(o) for o in outcomes], "table": table},
            f, indent=2,
        )
    return {"outcomes": [asdict(o) for o in outcomes]}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retraining strategy benchmark.")
    p.add_argument("--input", required=True)
    p.add_argument(
        "--windows", type=int, default=8,
        help="Number of chronological windows for replay.",
    )
    p.add_argument(
        "--cadence", type=int, default=3,
        help="Periodic cadence (windows) for the periodic / hybrid policies.",
    )
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(parse_args())
