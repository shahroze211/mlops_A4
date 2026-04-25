"""IEEE CIS - Cost-Sensitive Learning.

Trains a baseline LightGBM and a cost-sensitive variant whose loss is
weighted by both class imbalance *and* per-row TransactionAmt, then
projects the business impact: fraud loss avoided, false-alarm friction
cost, manual review cost, and net dollar delta.

Usage::

    python -m src.cost_sensitive --input outputs/features.parquet \
        --fraud-loss-ratio 1.0 --false-alarm-cost 5.0 --review-cost 2.0
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

LOG = logging.getLogger("cost_sensitive")

LABEL = "isFraud"
ID_COLS = ["TransactionID"]

# Default per-event business costs (USD).  Override via CLI.
DEFAULT_FRAUD_LOSS_RATIO = 1.0   # missed fraud loses the full transaction value
DEFAULT_FALSE_ALARM_COST = 5.0   # support / friction per false alarm
DEFAULT_REVIEW_COST = 2.0        # manual-review cost per flagged transaction


@dataclass
class BusinessImpact:
    strategy: str
    precision: float
    recall: float
    f1: float
    auc_roc: float
    fraud_loss_dollars: float
    false_alarm_cost: float
    review_cost: float
    total_cost: float
    captured_fraud_dollars: float
    confusion_matrix: List[List[int]]


def project_impact(
    name: str,
    y_true: pd.Series,
    preds: np.ndarray,
    probs: np.ndarray,
    amounts: pd.Series,
    fraud_loss_ratio: float,
    false_alarm_cost: float,
    review_cost: float,
) -> BusinessImpact:
    cm = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel()
    fraud_loss = float(amounts[(y_true == 1) & (preds == 0)].sum()) * fraud_loss_ratio
    captured = float(amounts[(y_true == 1) & (preds == 1)].sum()) * fraud_loss_ratio
    fa_cost = float(fp) * false_alarm_cost
    rev_cost = float(tp + fp) * review_cost
    return BusinessImpact(
        strategy=name,
        precision=float(precision_score(y_true, preds, zero_division=0)),
        recall=float(recall_score(y_true, preds, zero_division=0)),
        f1=float(f1_score(y_true, preds, zero_division=0)),
        auc_roc=float(roc_auc_score(y_true, probs)),
        fraud_loss_dollars=fraud_loss,
        false_alarm_cost=fa_cost,
        review_cost=rev_cost,
        total_cost=fraud_loss + fa_cost + rev_cost,
        captured_fraud_dollars=captured,
        confusion_matrix=cm.tolist(),
    )


def fit_baseline(X_tr, y_tr) -> LGBMClassifier:
    LOG.info("Fitting baseline LightGBM")
    clf = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=64,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    clf.fit(X_tr, y_tr)
    return clf


def fit_cost_sensitive(X_tr, y_tr, amt_tr: pd.Series) -> LGBMClassifier:
    """Class-weighted *and* sample-weighted by log(transaction_amount)."""
    LOG.info("Fitting cost-sensitive LightGBM (class_weight + amount-weighted)")
    sample_weight = np.where(
        y_tr.values == 1,
        np.log1p(amt_tr.clip(lower=1.0).values),
        1.0,
    )
    clf = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=64,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    clf.fit(X_tr, y_tr, sample_weight=sample_weight)
    return clf


def render_table(results: List[BusinessImpact]) -> str:
    header = (
        "strategy           prec    recall  f1      auc     "
        "fraud_loss     fa_cost     review     total_cost   captured"
    )
    lines = [header, "-" * len(header)]
    for r in results:
        lines.append(
            f"{r.strategy:<18} {r.precision:.4f}  {r.recall:.4f}  "
            f"{r.f1:.4f}  {r.auc_roc:.4f}  "
            f"${r.fraud_loss_dollars:>11,.0f}  ${r.false_alarm_cost:>9,.0f}  "
            f"${r.review_cost:>8,.0f}  ${r.total_cost:>10,.0f}   "
            f"${r.captured_fraud_dollars:>10,.0f}"
        )
    if len(results) >= 2:
        delta = results[0].total_cost - results[1].total_cost
        better = "cost_sensitive" if delta > 0 else "baseline"
        lines.append(
            f"\nNet delta (baseline - cost_sensitive) = ${delta:,.0f}  -> winner: {better}"
        )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> Dict:
    LOG.info("Loading %s", args.input)
    df = (
        pd.read_parquet(args.input)
        if args.input.endswith(".parquet")
        else pd.read_csv(args.input)
    )
    if LABEL not in df.columns or "TransactionAmt" not in df.columns:
        raise KeyError("Need both isFraud and TransactionAmt columns")
    y = df[LABEL].astype(int)
    amounts = df["TransactionAmt"].fillna(0).astype(float)
    X = df.drop(columns=[LABEL] + [c for c in ID_COLS if c in df.columns])
    X = X.select_dtypes(include=[np.number]).fillna(0)

    X_train, X_test, y_train, y_test, amt_tr, amt_te = train_test_split(
        X, y, amounts, test_size=0.2, stratify=y, random_state=42,
    )

    baseline = fit_baseline(X_train, y_train)
    cost = fit_cost_sensitive(X_train, y_train, amt_tr)

    results = [
        project_impact(
            "baseline",
            y_test,
            baseline.predict(X_test),
            baseline.predict_proba(X_test)[:, 1],
            amt_te,
            args.fraud_loss_ratio,
            args.false_alarm_cost,
            args.review_cost,
        ),
        project_impact(
            "cost_sensitive",
            y_test,
            cost.predict(X_test),
            cost.predict_proba(X_test)[:, 1],
            amt_te,
            args.fraud_loss_ratio,
            args.false_alarm_cost,
            args.review_cost,
        ),
    ]
    table = render_table(results)
    print(table)

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/cost_sensitive_report.json", "w") as f:
        json.dump(
            {"results": [asdict(r) for r in results], "table": table},
            f, indent=2,
        )
    return {"results": [asdict(r) for r in results]}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cost-sensitive training comparison.")
    p.add_argument("--input", required=True)
    p.add_argument(
        "--fraud-loss-ratio", type=float, default=DEFAULT_FRAUD_LOSS_RATIO,
        help="Fraction of TransactionAmt lost when a fraud is missed.",
    )
    p.add_argument(
        "--false-alarm-cost", type=float, default=DEFAULT_FALSE_ALARM_COST,
        help="Friction cost per false alarm (USD).",
    )
    p.add_argument(
        "--review-cost", type=float, default=DEFAULT_REVIEW_COST,
        help="Manual-review cost per flagged transaction (USD).",
    )
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(parse_args())
