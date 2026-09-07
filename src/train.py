"""Fit both models and write the artefacts the application loads.

Run from the repository root with ``python -m src.train``.

Two models are trained, for different jobs. The income classifier predicts which
side of the census threshold a person falls on. The intent classifier decides
what a user is asking. They share nothing but this script.

The income booster is written in XGBoost native JSON, which is designed to load
across library versions. The fitted scaler, the preparation schema and the
default field values are written together as one dictionary. Metadata records
the library versions, the hyperparameters and every metric, so that any figure
quoted elsewhere can be traced back to the run that produced it.

The intent classifier is fitted on the patterns in ``data/intents.json`` only.
The display strings shown on the interface buttons are deliberately excluded, so
that the accuracy reported for them describes the deployed model rather than one
that has already seen them.

The application never fits a model. It loads what this script writes.
"""

import json
import platform
from datetime import datetime, timezone

import joblib
import numpy as np
import sklearn
import xgboost as xgb
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score, f1_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler

from src import config
from src.data import load_raw, prepare_features, extract_target, compute_defaults
from src.nlp import build_vocabulary, vectorise_many

MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.1,
    "eval_metric": "logloss",
    "random_state": config.RANDOM_STATE,
}

VALIDATION_SIZE = 0.2

# Smoothing is lighter than the default of 1.0 because the patterns are hand
# written and sparse. All candidate values classify every display string
# correctly, so the choice was made on the confidence spread: 1.0 leaves half the
# buttons near 0.39, and 0.1 pushes the median above 0.96. Neither leaves room
# for a meaningful fallback threshold.
INTENT_ALPHA = 0.3

# Presence rather than counts. A question is short enough that a word rarely
# repeats, so counting adds noise without adding signal.
INTENT_BINARY = True


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


def train_income():
    """Fit the income classifier and write its artefacts to disk."""
    config.MODELS.mkdir(exist_ok=True)

    train_raw_all, test_raw = load_raw()

    # The raw records are split before any preparation rule is fitted. Rare
    # country grouping and the one hot category levels are learned from the
    # training rows alone and then reapplied unchanged, so validation rows
    # cannot influence the structure of the encoding.
    train_raw, val_raw = train_test_split(
        train_raw_all,
        test_size=VALIDATION_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=extract_target(train_raw_all),
    )

    X_train, schema = prepare_features(train_raw)
    X_val, _ = prepare_features(val_raw, schema)
    X_test, _ = prepare_features(test_raw, schema)

    y_train = extract_target(train_raw)
    y_val = extract_target(val_raw)
    y_test = extract_target(test_raw)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    model = xgb.XGBClassifier(**MODEL_PARAMS)
    model.fit(X_train_scaled, y_train)

    training_metrics = evaluate(model, X_train_scaled, y_train)
    validation_metrics = evaluate(model, X_val_scaled, y_val)
    test_metrics = evaluate(model, X_test_scaled, y_test)

    # Fitted on the training rows only, for the same reason as everything else.
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
        "feature_count": int(X_train.shape[1]),
        "positive_class_share_training_split": float(y_train.mean()),
        "confusion_matrix_held_out_test": confusion_matrix(
            y_test, model.predict(X_test_scaled)).tolist(),
        "metrics": {"training": training_metrics,
                    "validation": validation_metrics,
                    "held_out_test": test_metrics},
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


def load_intents():
    """Read the intent definitions, in file order."""
    with open(config.DATA / "intents.json", encoding="utf-8") as handle:
        return json.load(handle)["intents"]


def training_examples(intents):
    """Return the labelled patterns, excluding every display string."""
    documents, labels = [], []

    for intent in intents:
        display = intent["display"].lower().strip("?. ")
        for pattern in intent["patterns"]:
            if pattern.lower().strip("?. ") == display:
                raise ValueError(
                    f"intent '{intent['tag']}' lists its display string as a pattern, "
                    "which would make the button test meaningless")
            documents.append(pattern)
            labels.append(intent["tag"])

    return documents, labels


def train_intents():
    """Fit the intent classifier on the authored patterns and save it."""
    config.MODELS.mkdir(exist_ok=True)

    intents = load_intents()
    documents, labels = training_examples(intents)

    vocabulary = build_vocabulary(documents)
    X = vectorise_many(documents, vocabulary, binary=INTENT_BINARY)

    model = MultinomialNB(alpha=INTENT_ALPHA)
    model.fit(X, labels)

    tags = list(model.classes_)
    displays = [i["display"] for i in intents]
    expected = [i["tag"] for i in intents]

    probabilities = model.predict_proba(
        vectorise_many(displays, vocabulary, binary=INTENT_BINARY))
    predicted = [tags[i] for i in probabilities.argmax(axis=1)]
    confidences = probabilities.max(axis=1)
    correct = [p == e for p, e in zip(predicted, expected)]

    artefact = {
        "model": model,
        "vocabulary": vocabulary,
        "tags": tags,
        "alpha": INTENT_ALPHA,
        "binary": INTENT_BINARY,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "intent_count": len(intents),
        "pattern_count": len(documents),
        "vocabulary_size": len(vocabulary),
        "display_accuracy": float(np.mean(correct)),
        "lowest_display_confidence": float(confidences.min()),
        "median_display_confidence": float(np.median(confidences)),
        "display_failures": [
            {"display": d, "expected": e, "predicted": p, "confidence": float(c)}
            for d, e, p, c, ok in zip(displays, expected, predicted, confidences, correct)
            if not ok
        ],
        "versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
        },
    }

    joblib.dump(artefact, config.MODELS / "intent_model.joblib")

    return artefact


def report_income(metadata):
    """Print the income run in a form that can be checked against the notebook."""
    rows = metadata["rows"]
    print("Income classifier")
    print(f"  train {rows['train']}  validation {rows['validation']}  "
          f"held-out test {rows['held_out_test']}  features {metadata['feature_count']}")

    print(f"  {'split':<16}{'accuracy':>10}{'precision':>11}{'recall':>9}"
          f"{'f1':>9}{'roc_auc':>10}")
    for split, scores in metadata["metrics"].items():
        print(f"  {split:<16}"
              f"{scores['accuracy']:>10.4f}"
              f"{scores['precision']:>11.4f}"
              f"{scores['recall']:>9.4f}"
              f"{scores['f1']:>9.4f}"
              f"{scores['roc_auc']:>10.4f}")

    (true_lower, false_upper), (missed_upper, found_upper) = \
        metadata["confusion_matrix_held_out_test"]
    print(f"  held-out test: {found_upper} upper-bracket found, {missed_upper} missed, "
          f"{false_upper} wrongly flagged, {true_lower} lower-bracket correct")


def report_intents(artefact):
    """Print the intent run, including any button that fails to resolve."""
    print("Intent classifier")
    print(f"  {artefact['intent_count']} intents, {artefact['pattern_count']} patterns, "
          f"{artefact['vocabulary_size']} words after preprocessing")
    print(f"  alpha {artefact['alpha']}  binary presence {artefact['binary']}")
    print(f"  display strings resolving correctly {artefact['display_accuracy']:.1%}")
    print(f"  confidence on buttons: lowest {artefact['lowest_display_confidence']:.3f}, "
          f"median {artefact['median_display_confidence']:.3f}")

    for failure in artefact["display_failures"]:
        print(f"  FAILS  '{failure['display']}' reached {failure['predicted']} "
              f"instead of {failure['expected']} at {failure['confidence']:.3f}")


if __name__ == "__main__":
    report_income(train_income())
    print()
    report_intents(train_intents())

    versions = json.load(open(config.MODELS / "metadata.json"))["versions"]
    print(f"\nxgboost {versions['xgboost']}  scikit-learn {versions['scikit_learn']}")
    print(f"artefacts written to {config.MODELS}")
