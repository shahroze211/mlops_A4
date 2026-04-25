"""IEEE CIS - Multi-Model Training.

Fits XGBoost, LightGBM, and a Random-Forest + recursive feature
elimination hybrid. Reports precision, recall, F1, AUC-ROC, and the
confusion matrix for each model. Models and the JSON report are
persisted to ``outputs/``.

Usage::

    python -m src.train --input outputs/features.parquet
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

LOG = logging.getLogger("train")

LABEL = "isFraud"
ID_COLS = ["TransactionID"]


@dataclass
class ModelResult:
    name: str
    precision: float
    recall: float
    f1: float
    auc_roc: float
    confusion_matrix: List[List[int]]
    n_features: int
    selected_features: List[str]


def load_features(input_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    LOG.info("Loading features from %s", input_path)
    df = (
        pd.read_parquet(input_path)
        if input_path.endswith(".parquet")
        else pd.read_csv(input_path)
    )
    if LABEL not in df.columns:
        raise KeyError(f"'{LABEL}' label column not found")
    y = df[LABEL].astype(int)
    X = df.drop(columns=[LABEL] + [c for c in ID_COLS if c in df.columns])
    X = X.select_dtypes(include=[np.number]).fillna(0)
    LOG.info("Loaded %d rows, %d numeric features", len(X), X.shape[1])
    return X, y


def evaluate(
    name: str, clf, X_test, y_test, feature_names: List[str],
) -> ModelResult:
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, preds).tolist()
    return ModelResult(
        name=name,
        precision=float(precision_score(y_test, preds, zero_division=0)),
        recall=float(recall_score(y_test, preds, zero_division=0)),
        f1=float(f1_score(y_test, preds, zero_division=0)),
        auc_roc=float(roc_auc_score(y_test, probs)),
        confusion_matrix=cm,
        n_features=len(feature_names),
        selected_features=feature_names,
    )


def train_xgboost(
    X_tr, y_tr, X_te, y_te, feats: List[str],
) -> Tuple[XGBClassifier, ModelResult]:
    LOG.info("Fitting XGBoost (%d features)", len(feats))
    pos_rate = max(float(y_tr.mean()), 1e-6)
    clf = XGBClassifier(
        n_estimators=600,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        tree_method="hist",
        scale_pos_weight=(1 - pos_rate) / pos_rate,
        random_state=42,
        eval_metric="aucpr",
        n_jobs=-1,
    )
    clf.fit(X_tr, y_tr)
    return clf, evaluate("xgboost", clf, X_te, y_te, feats)


def train_lightgbm(
    X_tr, y_tr, X_te, y_te, feats: List[str],
) -> Tuple[LGBMClassifier, ModelResult]:
    LOG.info("Fitting LightGBM (%d features)", len(feats))
    clf = LGBMClassifier(
        n_estimators=600,
        num_leaves=64,
        learning_rate=0.05,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    clf.fit(X_tr, y_tr)
    return clf, evaluate("lightgbm", clf, X_te, y_te, feats)


def train_rf_with_selection(
    X_tr, y_tr, X_te, y_te, feats: List[str],
) -> Tuple[RandomForestClassifier, ModelResult]:
    """Hybrid: RFECV feature selection on a small RF, then full RF re-fit."""
    LOG.info("Hybrid: RFECV(RF) + RandomForest re-fit on selected subset")
    base = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42,
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    selector = RFECV(
        estimator=base,
        step=0.2,
        min_features_to_select=max(10, int(len(feats) * 0.4)),
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
    )
    if isinstance(X_tr, np.ndarray):
        X_tr_df = pd.DataFrame(X_tr, columns=feats)
        X_te_df = pd.DataFrame(X_te, columns=feats)
    else:
        X_tr_df, X_te_df = X_tr, X_te

    selector.fit(X_tr_df, y_tr)
    keep_mask = selector.support_
    keep = [f for f, m in zip(feats, keep_mask) if m]
    LOG.info("RFECV kept %d / %d features", len(keep), len(feats))
    X_tr_sel = X_tr_df.loc[:, keep]
    X_te_sel = X_te_df.loc[:, keep]

    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=14,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X_tr_sel, y_tr)
    return clf, evaluate("random_forest_rfe", clf, X_te_sel, y_te, keep)


def render_table(results: List[ModelResult]) -> str:
    lines = [
        "model               precision  recall    f1        auc       n_features",
        "-" * 80,
    ]
    for r in results:
        lines.append(
            f"{r.name:<18}  {r.precision:.4f}     {r.recall:.4f}    "
            f"{r.f1:.4f}    {r.auc_roc:.4f}    {r.n_features}"
        )
    lines.append("\nConfusion matrices [[tn, fp], [fn, tp]]:")
    for r in results:
        lines.append(f"  {r.name}: {r.confusion_matrix}")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> Dict:
    X, y = load_features(args.input)
    feats = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42,
    )
    scaler = StandardScaler(with_mean=False)
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    os.makedirs("outputs", exist_ok=True)
    results: List[ModelResult] = []
    for fn in (train_xgboost, train_lightgbm, train_rf_with_selection):
        model, result = fn(X_train_s, y_train, X_test_s, y_test, feats)
        results.append(result)
        joblib.dump(
            {"model": model, "features": result.selected_features},
            f"outputs/{result.name}.joblib",
        )

    table = render_table(results)
    print(table)

    with open("outputs/training_report.json", "w") as f:
        json.dump({"results": [asdict(r) for r in results]}, f, indent=2)

    # Also drop a live_metrics.json for the Prometheus exporter to pick up.
    best = max(results, key=lambda r: r.auc_roc)
    cm = np.array(best.confusion_matrix)
    fp_rate = cm[0, 1] / max(cm[0].sum(), 1)
    with open("outputs/live_metrics.json", "w") as f:
        json.dump(
            {
                "recall": best.recall,
                "precision": best.precision,
                "f1": best.f1,
                "auc_roc": best.auc_roc,
                "false_positive_rate": float(fp_rate),
                "best_model": best.name,
            },
            f,
            indent=2,
        )
    return {"results": [asdict(r) for r in results]}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-model fraud training.")
    p.add_argument(
        "--input", required=True,
        help="Engineered features (parquet/csv) with isFraud column.",
    )
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(parse_args())
