"""IEEE CIS - Explainability artefacts.

Trains LightGBM on the supplied features (or loads a pre-trained model
bundle), then produces:

  * ``outputs/shap_summary.png``      - SHAP beeswarm summary plot
  * ``outputs/feature_importance.png`` - top-N gain bar chart

Usage::

    python -m src.explainability --input outputs/features.parquet
    python -m src.explainability --input outputs/features.parquet \
        --model outputs/lightgbm.joblib
"""
from __future__ import annotations

import argparse
import logging
import os
from typing import List, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

LOG = logging.getLogger("explainability")

LABEL = "isFraud"
ID_COLS = ["TransactionID"]


def load_data(path: str) -> Tuple[pd.DataFrame, pd.Series]:
    LOG.info("Loading %s", path)
    df = pd.read_csv(path) if path.endswith(".csv") else pd.read_parquet(path)
    if LABEL not in df.columns:
        raise KeyError(f"'{LABEL}' column not found in {path}")
    y = df[LABEL].astype(int)
    X = df.drop(columns=[LABEL] + [c for c in ID_COLS if c in df.columns])
    X = X.select_dtypes(include=[np.number]).fillna(0)
    return X, y


def fit_or_load(
    X: pd.DataFrame, y: pd.Series, model_path: str | None,
) -> Tuple[LGBMClassifier, List[str]]:
    if model_path and os.path.exists(model_path):
        LOG.info("Loading model bundle from %s", model_path)
        bundle = joblib.load(model_path)
        if isinstance(bundle, dict):
            model = bundle["model"]
            feats = bundle.get("features") or list(X.columns)
        else:
            model = bundle
            feats = list(X.columns)
        return model, feats
    LOG.info("No pre-trained model supplied; fitting LightGBM on the input data")
    X_tr, _, y_tr, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42,
    )
    clf = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=64,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    clf.fit(X_tr, y_tr)
    return clf, list(X.columns)


def shap_summary(
    clf: LGBMClassifier, X: pd.DataFrame, feats: List[str], sample: int, out_path: str,
) -> None:
    if len(X) > sample:
        X = X.sample(sample, random_state=42)
    X = X.reindex(columns=feats).fillna(0)
    LOG.info("Computing SHAP values on %d rows", len(X))
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        # Binary classifiers return a list per class - take the positive class.
        shap_values = shap_values[1]
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X, max_display=25, show=False, plot_size=(10, 8),
    )
    plt.title("SHAP Summary - IEEE CIS Fraud Detection")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    LOG.info("Saved SHAP summary -> %s", out_path)


def importance_bar(
    clf: LGBMClassifier, feature_names: List[str], out_path: str, top: int = 30,
) -> None:
    importances = clf.booster_.feature_importance(importance_type="gain")
    df = (
        pd.DataFrame({"feature": feature_names, "gain": importances})
        .sort_values("gain", ascending=False)
        .head(top)
        .iloc[::-1]
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(df["feature"], df["gain"], color="#1f77b4", edgecolor="white")
    ax.set_xlabel("Gain")
    ax.set_title(f"Top {top} feature importances - IEEE CIS Fraud")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    LOG.info("Saved feature importance -> %s", out_path)


def run(args: argparse.Namespace) -> None:
    X, y = load_data(args.input)
    clf, feats = fit_or_load(X, y, args.model)
    os.makedirs("outputs", exist_ok=True)
    shap_summary(clf, X, feats, args.shap_sample, "outputs/shap_summary.png")
    importance_bar(clf, feats, "outputs/feature_importance.png")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SHAP + feature importance plots.")
    p.add_argument("--input", required=True, help="Features (parquet/csv).")
    p.add_argument(
        "--model", default=None,
        help="Optional joblib model bundle (uses pre-trained model if provided).",
    )
    p.add_argument(
        "--shap-sample", type=int, default=2000,
        help="Number of rows to subsample for the SHAP computation.",
    )
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(parse_args())
