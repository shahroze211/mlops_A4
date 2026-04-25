"""IEEE CIS - Data Preprocessing.

Handles missing values, target-encodes high-cardinality columns, and
benchmarks SMOTE oversampling against ``class_weight='balanced'`` on a
held-out slice. Outputs a side-by-side metrics table to stdout and to
``outputs/preprocessing_comparison.json``.

Usage::

    python -m src.data_preprocessing --input train_transaction.csv [--sample 100000]
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
from category_encoders import TargetEncoder
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

LOG = logging.getLogger("preprocess")

# IEEE CIS column groupings.
HIGH_CARD_COLS = ["card1", "card2", "card3", "card5", "addr1", "addr2"]
LOW_CARD_CAT_COLS = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain"]
ID_COLS = ["TransactionID"]
LABEL = "isFraud"


@dataclass
class StrategyResult:
    strategy: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float
    fraud_caught: int
    false_alarms: int
    legit_correct: int
    fraud_missed: int


def load_dataset(path: str, sample: int | None = None) -> pd.DataFrame:
    LOG.info("Loading %s", path)
    df = pd.read_csv(path) if path.endswith(".csv") else pd.read_parquet(path)
    if LABEL not in df.columns:
        raise KeyError(f"Expected '{LABEL}' column not found in {path}")
    if sample is not None and len(df) > sample:
        df = df.sample(sample, random_state=42).reset_index(drop=True)
        LOG.info("Down-sampled to %d rows", sample)
    return df


def split_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    cat_low = [c for c in LOW_CARD_CAT_COLS if c in df.columns]
    cat_high = [c for c in HIGH_CARD_COLS if c in df.columns]
    excluded = set(cat_low + cat_high + ID_COLS + [LABEL])
    num_cols = [
        c for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]
    LOG.info(
        "Column split: %d numeric, %d low-card cat, %d high-card cat",
        len(num_cols), len(cat_low), len(cat_high),
    )
    return num_cols, cat_low, cat_high


def handle_missing_values(
    df: pd.DataFrame, num_cols: List[str], cat_cols: List[str],
) -> pd.DataFrame:
    """Median impute numerics, ``"missing"`` token for categoricals."""
    df = df.copy()
    if num_cols:
        df[num_cols] = df[num_cols].astype(float)
        medians = df[num_cols].median(numeric_only=True)
        df[num_cols] = df[num_cols].fillna(medians)
    for c in cat_cols:
        df[c] = df[c].fillna("missing").astype(str)
    return df


def target_encode_high_cardinality(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, TargetEncoder | None]:
    """Smoothing-regularised out-of-fold target encoding."""
    if not cols:
        return X_train, X_test, None
    enc = TargetEncoder(cols=cols, smoothing=20.0, min_samples_leaf=20)
    X_train_enc = enc.fit_transform(X_train, y_train)
    X_test_enc = enc.transform(X_test)
    LOG.info("Target-encoded %d high-cardinality columns", len(cols))
    return X_train_enc, X_test_enc, enc


def evaluate(strategy: str, model, X_test, y_test) -> StrategyResult:
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    return StrategyResult(
        strategy=strategy,
        accuracy=float(accuracy_score(y_test, preds)),
        precision=float(precision_score(y_test, preds, zero_division=0)),
        recall=float(recall_score(y_test, preds, zero_division=0)),
        f1=float(f1_score(y_test, preds, zero_division=0)),
        auc_roc=float(roc_auc_score(y_test, probs)),
        fraud_caught=int(tp),
        false_alarms=int(fp),
        legit_correct=int(tn),
        fraud_missed=int(fn),
    )


def train_smote(X_train, y_train, X_test, y_test) -> StrategyResult:
    LOG.info("Strategy A: SMOTE oversample -> LogisticRegression")
    pos_count = int(np.sum(y_train == 1))
    k_neighbors = max(1, min(5, pos_count - 1))
    smote = SMOTE(sampling_strategy=0.5, random_state=42, k_neighbors=k_neighbors)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    LOG.info(
        "SMOTE resampled %d -> %d (positives now %d)",
        len(y_train), len(y_res), int(np.sum(y_res == 1)),
    )
    clf = LogisticRegression(max_iter=2000, n_jobs=-1, random_state=42)
    clf.fit(X_res, y_res)
    return evaluate("SMOTE", clf, X_test, y_test)


def train_class_weight(X_train, y_train, X_test, y_test) -> StrategyResult:
    LOG.info("Strategy B: class_weight='balanced' -> LogisticRegression")
    clf = LogisticRegression(
        max_iter=2000, n_jobs=-1, random_state=42, class_weight="balanced",
    )
    clf.fit(X_train, y_train)
    return evaluate("class_weight", clf, X_test, y_test)


def render_comparison(results: List[StrategyResult]) -> str:
    lines = [
        "strategy        acc      prec     recall   f1       auc      caught  missed   fp",
        "-" * 90,
    ]
    for r in results:
        lines.append(
            f"{r.strategy:<14}  {r.accuracy:.4f}   {r.precision:.4f}   "
            f"{r.recall:.4f}   {r.f1:.4f}   {r.auc_roc:.4f}   "
            f"{r.fraud_caught:>6}  {r.fraud_missed:>6}  {r.false_alarms:>6}"
        )
    if len(results) >= 2:
        lift = results[0].recall - results[1].recall
        winner = results[0].strategy if lift > 0 else results[1].strategy
        lines.append(
            f"\nRecall delta (SMOTE - class_weight) = {lift:+.4f} -> winner: {winner}"
        )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> Dict:
    df = load_dataset(args.input, sample=args.sample)
    num_cols, cat_low, cat_high = split_columns(df)
    df = handle_missing_values(df, num_cols, cat_low + cat_high)

    y = df[LABEL].astype(int)
    X = df.drop(columns=[LABEL] + [c for c in ID_COLS if c in df.columns])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42,
    )

    X_train, X_test, _ = target_encode_high_cardinality(
        X_train, X_test, y_train, cat_high + cat_low,
    )

    X_train = X_train.select_dtypes(include=[np.number]).fillna(0)
    X_test = X_test.select_dtypes(include=[np.number]).fillna(0)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = [
        train_smote(X_train_s, y_train, X_test_s, y_test),
        train_class_weight(X_train_s, y_train, X_test_s, y_test),
    ]
    table = render_comparison(results)
    print(table)

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/preprocessing_comparison.json", "w") as f:
        json.dump(
            {"results": [asdict(r) for r in results], "table": table},
            f, indent=2,
        )
    return {"results": [asdict(r) for r in results]}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IEEE CIS preprocessing comparison.")
    p.add_argument("--input", required=True, help="Path to train_transaction.csv")
    p.add_argument(
        "--sample", type=int, default=None,
        help="Optional row cap for fast iteration.",
    )
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(parse_args())
