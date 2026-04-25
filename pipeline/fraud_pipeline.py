"""IEEE CIS Fraud Detection - Kubeflow Pipeline.

Eight-component end-to-end pipeline.  Every task is wrapped with a retry
decorator and the deploy step is gated by a configurable accuracy
threshold via ``dsl.Condition``.

Components:
    1. ingest_data
    2. validate_data
    3. preprocess_data
    4. engineer_features
    5. tune_hyperparameters
    6. train_model
    7. evaluate_model
    8. conditional_deploy_model

Compile with::

    python pipeline/fraud_pipeline.py
    # produces fraud_pipeline.yaml in the current directory
"""
import logging
from typing import NamedTuple

from kfp import compiler, dsl
from kfp.dsl import (
    Artifact,
    ClassificationMetrics,
    Dataset,
    Input,
    Metrics,
    Model,
    Output,
    component,
    pipeline,
)

PIPELINE_IMAGE = "python:3.10-slim"

PD_PKGS = ["pandas==2.1.4", "numpy==1.26.4", "pyarrow==14.0.2"]
SK_PKGS = PD_PKGS + ["scikit-learn==1.4.0", "scipy==1.11.4"]
TRAIN_PKGS = SK_PKGS + [
    "xgboost==2.0.3",
    "lightgbm==4.2.0",
    "imbalanced-learn==0.12.0",
    "category-encoders==2.6.3",
    "joblib==1.3.2",
]


# =====================================================================
# Component 1: Ingest
# =====================================================================
@component(base_image=PIPELINE_IMAGE, packages_to_install=PD_PKGS + ["fsspec==2023.12.2"])
def ingest_data(raw_csv_uri: str, output_dataset: Output[Dataset]) -> None:
    """Pull raw IEEE CIS transaction CSV into the pipeline volume."""
    import logging

    import pandas as pd

    log = logging.getLogger("ingest")
    log.setLevel(logging.INFO)
    log.info("Reading raw fraud CSV from %s", raw_csv_uri)
    df = pd.read_csv(raw_csv_uri)
    required = {"TransactionID", "isFraud", "TransactionAmt"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Raw data missing required columns: {missing}")
    log.info("Ingested %d rows, %d cols", len(df), df.shape[1])
    df.to_parquet(output_dataset.path, index=False)
    output_dataset.metadata["rows"] = len(df)
    output_dataset.metadata["columns"] = df.shape[1]
    output_dataset.metadata["fraud_rate"] = float(df["isFraud"].mean())


# =====================================================================
# Component 2: Validate
# =====================================================================
@component(base_image=PIPELINE_IMAGE, packages_to_install=PD_PKGS)
def validate_data(input_dataset: Input[Dataset], report: Output[Artifact]) -> bool:
    """Schema, null-rate, and target-balance validation."""
    import json

    import pandas as pd

    df = pd.read_parquet(input_dataset.path)
    checks = {
        "rows": int(len(df)),
        "label_present": "isFraud" in df.columns,
        "label_binary": bool(set(df["isFraud"].dropna().unique()).issubset({0, 1})),
        "amount_non_negative": bool((df["TransactionAmt"].dropna() >= 0).all()),
        "fraud_rate": float(df["isFraud"].mean()),
        "max_null_rate": float(df.isna().mean().max()),
    }
    checks["passed"] = bool(
        checks["rows"] >= 1000
        and checks["label_present"]
        and checks["label_binary"]
        and checks["amount_non_negative"]
        and checks["fraud_rate"] > 0
    )
    with open(report.path, "w") as f:
        json.dump(checks, f, indent=2)
    if not checks["passed"]:
        raise RuntimeError(f"Validation failed: {checks}")
    return True


# =====================================================================
# Component 3: Preprocess
# =====================================================================
@component(base_image=PIPELINE_IMAGE, packages_to_install=TRAIN_PKGS)
def preprocess_data(
    input_dataset: Input[Dataset],
    output_train: Output[Dataset],
    output_test: Output[Dataset],
    test_size: float = 0.2,
) -> None:
    """Impute missing, target-encode high-cardinality, stratified split."""
    import pandas as pd
    from category_encoders import TargetEncoder
    from sklearn.model_selection import train_test_split

    df = pd.read_parquet(input_dataset.path)
    y = df["isFraud"].astype(int)
    X = df.drop(columns=["isFraud"])

    cat_low = [c for c in ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain"]
               if c in X.columns]
    cat_high = [c for c in ["card1", "card2", "card3", "card5", "addr1", "addr2"]
                if c in X.columns]
    num_cols = [c for c in X.columns
                if c not in cat_low + cat_high + ["TransactionID"]
                and pd.api.types.is_numeric_dtype(X[c])]

    if num_cols:
        X[num_cols] = X[num_cols].astype(float)
        X[num_cols] = X[num_cols].fillna(X[num_cols].median(numeric_only=True))
    for c in cat_low + cat_high:
        X[c] = X[c].fillna("missing").astype(str)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42,
    )
    if cat_low + cat_high:
        enc = TargetEncoder(cols=cat_low + cat_high, smoothing=20.0)
        X_tr = enc.fit_transform(X_tr, y_tr)
        X_te = enc.transform(X_te)
    pd.concat([X_tr, y_tr.rename("isFraud")], axis=1).to_parquet(
        output_train.path, index=False,
    )
    pd.concat([X_te, y_te.rename("isFraud")], axis=1).to_parquet(
        output_test.path, index=False,
    )
    output_train.metadata["rows"] = int(len(X_tr))
    output_test.metadata["rows"] = int(len(X_te))


# =====================================================================
# Component 4: Engineer features
# =====================================================================
@component(base_image=PIPELINE_IMAGE, packages_to_install=PD_PKGS)
def engineer_features(
    train_in: Input[Dataset],
    test_in: Input[Dataset],
    train_out: Output[Dataset],
    test_out: Output[Dataset],
) -> None:
    """Time-of-day, log amount, V-block aggregates."""
    import numpy as np
    import pandas as pd

    def add_feats(df: pd.DataFrame) -> pd.DataFrame:
        if "TransactionDT" in df.columns:
            df["hour_of_day"] = (df["TransactionDT"] // 3600) % 24
            df["day_of_week"] = (df["TransactionDT"] // 86400) % 7
        if "TransactionAmt" in df.columns:
            df["log_amount"] = np.log1p(df["TransactionAmt"].clip(lower=0))
        v_cols = [c for c in df.columns if c.startswith("V") and c[1:].isdigit()]
        if v_cols:
            v_block = df[v_cols]
            df["V_mean"] = v_block.mean(axis=1)
            df["V_std"] = v_block.std(axis=1)
            df["V_null_count"] = v_block.isna().sum(axis=1)
        return df

    tr = add_feats(pd.read_parquet(train_in.path))
    te = add_feats(pd.read_parquet(test_in.path))
    tr.to_parquet(train_out.path, index=False)
    te.to_parquet(test_out.path, index=False)


# =====================================================================
# Component 5: Hyperparameter tuning
# =====================================================================
@component(base_image=PIPELINE_IMAGE, packages_to_install=TRAIN_PKGS)
def tune_hyperparameters(
    train_in: Input[Dataset],
    best_params: Output[Artifact],
    n_iter: int = 10,
) -> None:
    """Randomised search over LightGBM hyperparameter space."""
    import json

    import numpy as np
    import pandas as pd
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

    df = pd.read_parquet(train_in.path).fillna(0)
    y = df["isFraud"].astype(int)
    X = df.drop(columns=["isFraud", "TransactionID"], errors="ignore")
    X = X.select_dtypes(include=[np.number])

    space = {
        "num_leaves": [31, 63, 127],
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [200, 400, 600],
        "min_child_samples": [20, 50, 100],
        "reg_alpha": [0.0, 0.1, 1.0],
        "feature_fraction": [0.7, 0.85, 1.0],
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        LGBMClassifier(class_weight="balanced", random_state=42, verbose=-1),
        space,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X, y)
    payload = {
        "best_params": search.best_params_,
        "best_score": float(search.best_score_),
        "n_iter": n_iter,
    }
    with open(best_params.path, "w") as f:
        json.dump(payload, f, indent=2)


# =====================================================================
# Component 6: Train
# =====================================================================
@component(base_image=PIPELINE_IMAGE, packages_to_install=TRAIN_PKGS)
def train_model(
    train_in: Input[Dataset],
    best_params: Input[Artifact],
    model_out: Output[Model],
) -> None:
    """Fit LightGBM with the tuned hyperparameters."""
    import json

    import joblib
    import numpy as np
    import pandas as pd
    from lightgbm import LGBMClassifier

    df = pd.read_parquet(train_in.path).fillna(0)
    y = df["isFraud"].astype(int)
    X = df.drop(columns=["isFraud", "TransactionID"], errors="ignore")
    X = X.select_dtypes(include=[np.number])

    with open(best_params.path) as f:
        params = json.load(f)["best_params"]

    clf = LGBMClassifier(
        class_weight="balanced", random_state=42, verbose=-1, **params,
    )
    clf.fit(X, y)
    joblib.dump(
        {"model": clf, "feature_names": list(X.columns)},
        model_out.path,
    )
    model_out.metadata["framework"] = "lightgbm"
    model_out.metadata["features"] = int(len(X.columns))
    model_out.metadata["params"] = json.dumps(params)


# =====================================================================
# Component 7: Evaluate (returns floats for downstream gating)
# =====================================================================
@component(base_image=PIPELINE_IMAGE, packages_to_install=TRAIN_PKGS)
def evaluate_model(
    test_in: Input[Dataset],
    model_in: Input[Model],
    metrics_out: Output[Metrics],
    cls_metrics_out: Output[ClassificationMetrics],
) -> NamedTuple("EvalOut", [("accuracy", float), ("auc", float), ("recall", float)]):
    """Score the holdout slice and emit metrics + a NamedTuple for the gate."""
    from collections import namedtuple

    import joblib
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score, confusion_matrix, f1_score, precision_score,
        recall_score, roc_auc_score, roc_curve,
    )

    bundle = joblib.load(model_in.path)
    clf = bundle["model"]
    feats = bundle["feature_names"]

    df = pd.read_parquet(test_in.path).fillna(0)
    y = df["isFraud"].astype(int)
    X = df.reindex(columns=feats).fillna(0)
    preds = clf.predict(X)
    probs = clf.predict_proba(X)[:, 1]

    acc = float(accuracy_score(y, preds))
    auc = float(roc_auc_score(y, probs))
    rec = float(recall_score(y, preds))
    prec = float(precision_score(y, preds))
    f1 = float(f1_score(y, preds))

    metrics_out.log_metric("accuracy", acc)
    metrics_out.log_metric("precision", prec)
    metrics_out.log_metric("recall", rec)
    metrics_out.log_metric("f1", f1)
    metrics_out.log_metric("auc_roc", auc)

    cm = confusion_matrix(y, preds).tolist()
    cls_metrics_out.log_confusion_matrix(["legit", "fraud"], cm)
    fpr, tpr, _ = roc_curve(y, probs)
    cls_metrics_out.log_roc_curve(fpr.tolist(), tpr.tolist(), [0.0] * len(fpr))

    EvalOut = namedtuple("EvalOut", ["accuracy", "auc", "recall"])
    return EvalOut(acc, auc, rec)


# =====================================================================
# Component 8: Conditional deploy
# =====================================================================
@component(base_image=PIPELINE_IMAGE, packages_to_install=PD_PKGS)
def conditional_deploy_model(
    model_in: Input[Model],
    accuracy: float,
    auc: float,
    recall: float,
    threshold: float,
    deployment_target: str,
    deploy_record: Output[Artifact],
) -> str:
    """Promote to staging if metrics clear gate, else log skip reason."""
    import json
    from datetime import datetime, timezone

    decision = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_uri": model_in.uri,
        "accuracy": accuracy,
        "auc": auc,
        "recall": recall,
        "threshold": threshold,
        "target": deployment_target,
    }
    if accuracy >= threshold and auc >= 0.85:
        decision["action"] = "promoted"
        decision["served_at"] = f"{deployment_target}/fraud-detector"
    else:
        decision["action"] = "skipped"
        decision["reason"] = (
            f"accuracy {accuracy:.3f} < {threshold:.3f} or auc {auc:.3f} < 0.85"
        )
    with open(deploy_record.path, "w") as f:
        json.dump(decision, f, indent=2)
    return decision["action"]


# =====================================================================
# Pipeline definition
# =====================================================================
@pipeline(
    name="ieee-cis-fraud-detection",
    description="End-to-end Kubeflow pipeline for IEEE CIS fraud detection",
)
def fraud_pipeline(
    raw_csv_uri: str = "gs://fraud-data/ieee-cis/train_transaction.csv",
    accuracy_threshold: float = 0.92,
    deployment_target: str = "kserve://fraud-detection",
    tune_iterations: int = 10,
):
    ingest = ingest_data(raw_csv_uri=raw_csv_uri)
    ingest.set_retry(num_retries=3, backoff_duration="60s", backoff_factor=2)
    ingest.set_caching_options(False)

    validate = validate_data(input_dataset=ingest.outputs["output_dataset"])
    validate.set_retry(num_retries=2, backoff_duration="30s")

    preprocess = preprocess_data(
        input_dataset=ingest.outputs["output_dataset"],
    ).after(validate)
    preprocess.set_retry(num_retries=2, backoff_duration="60s")

    feats = engineer_features(
        train_in=preprocess.outputs["output_train"],
        test_in=preprocess.outputs["output_test"],
    )
    feats.set_retry(num_retries=2, backoff_duration="30s")

    tune = tune_hyperparameters(
        train_in=feats.outputs["train_out"],
        n_iter=tune_iterations,
    )
    tune.set_retry(num_retries=3, backoff_duration="120s")

    trained = train_model(
        train_in=feats.outputs["train_out"],
        best_params=tune.outputs["best_params"],
    )
    trained.set_retry(num_retries=3, backoff_duration="120s")

    evaluated = evaluate_model(
        test_in=feats.outputs["test_out"],
        model_in=trained.outputs["model_out"],
    )
    evaluated.set_retry(num_retries=2, backoff_duration="30s")

    with dsl.Condition(
        evaluated.outputs["accuracy"] >= accuracy_threshold,
        name="accuracy-gate",
    ):
        deploy = conditional_deploy_model(
            model_in=trained.outputs["model_out"],
            accuracy=evaluated.outputs["accuracy"],
            auc=evaluated.outputs["auc"],
            recall=evaluated.outputs["recall"],
            threshold=accuracy_threshold,
            deployment_target=deployment_target,
        )
        deploy.set_retry(num_retries=3, backoff_duration="60s")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    compiler.Compiler().compile(
        pipeline_func=fraud_pipeline,
        package_path="fraud_pipeline.yaml",
    )
    logging.getLogger().info("Compiled fraud_pipeline.yaml")
