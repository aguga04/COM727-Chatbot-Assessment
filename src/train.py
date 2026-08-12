"""Fit the income classifier and write the artefacts the application loads.

Run from the repository root with ``python -m src.train``.

Three files are produced in ``models/``. The trained booster is written in
XGBoost native JSON, which is designed to load across library versions. The
fitted scaler, the preparation schema and the default field values are written
together as one dictionary. Metadata records the library versions, the
hyperparameters and every metric, so that any figure quoted elsewhere can be
traced back to the run that produced it.

The application never fits a model. It loads what this script writes.
"""

import json
import platform
from datetime import datetime, timezone

import joblib
import numpy as np
import sklearn
import xgboost as xgb
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src import config
from src.data import load_raw, prepare_features, extract_target, compute_defaults

MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.1,
    "eval_metric": "logloss",
    "random_state": config.RANDOM_STATE,
}

VALIDATION_SIZE = 0.2


def evaluate(model, X, y):
    """Return the standard classification metrics as a dictionary."""
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions)),
        "recall": float(recall_score(y, predictions)),
        "f1": float(f1_score(y, predictions)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
    }


def train():
    """Fit the model on the training split and write all artefacts to disk."""
    config.MODELS.mkdir(exist_ok=True)

    train_raw, test_raw = load_raw()

    X, schema = prepare_features(train_raw)
    X_test, _ = prepare_features(test_raw, schema)
    y = extract_target(train_raw)
    y_test = extract_target(test_raw)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=VALIDATION_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    model = xgb.XGBClassifier(**MODEL_PARAMS)
    model.fit(X_train_scaled, y_train)

    validation_metrics = evaluate(model, X_val_scaled, y_val)
    test_metrics = evaluate(model, X_test_scaled, y_test)

    defaults = compute_defaults(train_raw)

    model.get_booster().save_model(config.MODELS / "income_model.json")

    joblib.dump(
        {"scaler": scaler, "schema": schema, "defaults": defaults},
        config.MODELS / "preprocessing.joblib",
    )

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": "XGBClassifier",
        "hyperparameters": MODEL_PARAMS,
        "validation_size": VALIDATION_SIZE,
        "random_state": config.RANDOM_STATE,
        "rows": {
            "train": int(X_train.shape[0]),
            "validation": int(X_val.shape[0]),
            "held_out_test": int(X_test.shape[0]),
        },
        "feature_count": int(X.shape[1]),
        "positive_class_share": float(y.mean()),
        "metrics": {"validation": validation_metrics, "held_out_test": test_metrics},
        "versions": {
            "python": platform.python_version(),
            "xgboost": xgb.__version__,
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
    }

    with open(config.MODELS / "metadata.json", "w") as handle:
        json.dump(metadata, handle, indent=2)

    return metadata


def report(metadata):
    """Print the run summary in a form that can be checked against the notebook."""
    rows = metadata["rows"]
    print(f"train {rows['train']}  validation {rows['validation']}  "
          f"held-out test {rows['held_out_test']}  features {metadata['feature_count']}")

    header = f"{'split':<16}{'accuracy':>10}{'precision':>11}{'recall':>9}{'f1':>9}{'roc_auc':>10}"
    print(header)
    for split, scores in metadata["metrics"].items():
        print(f"{split:<16}"
              f"{scores['accuracy']:>10.4f}"
              f"{scores['precision']:>11.4f}"
              f"{scores['recall']:>9.4f}"
              f"{scores['f1']:>9.4f}"
              f"{scores['roc_auc']:>10.4f}")

    versions = metadata["versions"]
    print(f"xgboost {versions['xgboost']}  scikit-learn {versions['scikit_learn']}")
    print(f"artefacts written to {config.MODELS}")


if __name__ == "__main__":
    report(train())