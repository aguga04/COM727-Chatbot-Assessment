"""Load the trained artefacts and score individual records.

The application does not fit a model. This module reads what ``src.train``
wrote and applies it with the preparation the model was trained on.

Artefacts are loaded once and cached at module level.
"""

import json

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb

from src import config
from src.data import prepare_features, build_person

_artefacts = None


def load_artefacts():
    """Return the trained booster, the preprocessing objects and the metadata.

    The result is cached after the first call. Keys are ``booster``, ``scaler``,
    ``schema``, ``defaults`` and ``metadata``.
    """
    global _artefacts

    if _artefacts is None:
        booster = xgb.Booster()
        booster.load_model(config.MODELS / "income_model.json")

        preprocessing = joblib.load(config.MODELS / "preprocessing.joblib")

        with open(config.MODELS / "metadata.json") as handle:
            metadata = json.load(handle)

        _artefacts = {
            "booster": booster,
            "scaler": preprocessing["scaler"],
            "schema": preprocessing["schema"],
            "defaults": preprocessing["defaults"],
            "metadata": metadata,
        }

    return _artefacts


def version_report():
    """Compare the libraries in use against those recorded at training time.

    Returns a list of warnings, empty when everything matches. The scaler is a
    pickled object, so a scikit-learn mismatch can alter its behaviour without
    raising.
    """
    recorded = load_artefacts()["metadata"]["versions"]
    installed = {"xgboost": xgb.__version__, "scikit_learn": sklearn.__version__}

    warnings = []
    for library, current in installed.items():
        expected = recorded.get(library)
        if expected is not None and expected != current:
            warnings.append(
                f"{library} {current} is in use but the model was trained with {expected}"
            )
    return warnings


def complete_record(answers):
    """Complete a partial set of answers.

    Years of education are derived from the qualification, since the two record
    the same attribute. Anything else absent falls back to the training median
    or mode.

    Returns a single-row DataFrame and the sorted list of defaulted fields.
    """
    defaults = load_artefacts()["defaults"]
    supplied = {k: v for k, v in answers.items() if v is not None}

    education = supplied.get("education")
    if education in config.EDUCATION_NUM and "education-num" not in supplied:
        supplied["education-num"] = config.EDUCATION_NUM[education]

    return build_person(supplied, defaults)


def prepare_matrix(df_raw):
    """Turn raw records into the scaled feature matrix the booster expects.

    Applies the fitted encoding then the fitted scaler. Returns the matrix and
    the ordered feature names.
    """
    artefacts = load_artefacts()
    features, _ = prepare_features(df_raw, artefacts["schema"])
    scaled = artefacts["scaler"].transform(features)
    return scaled, artefacts["schema"]["feature_columns"]


def predict_probabilities(df_raw):
    """Return the predicted probability of the upper income bracket per row."""
    scaled, _ = prepare_matrix(df_raw)
    booster = load_artefacts()["booster"]
    return booster.predict(xgb.DMatrix(scaled))


def score_person(answers):
    """Score one record described as a dictionary of answers.

    Returns the predicted probability, the completed record, the fields that
    took a default, and the scaled feature row for the explanation module.
    """
    record, defaulted = complete_record(answers)
    scaled, feature_names = prepare_matrix(record)
    booster = load_artefacts()["booster"]

    probability = float(booster.predict(xgb.DMatrix(scaled))[0])

    return {
        "probability": probability,
        "record": record,
        "defaulted": defaulted,
        "scaled": scaled,
        "feature_names": feature_names,
    }


if __name__ == "__main__":
    for warning in version_report():
        print(f"warning: {warning}")

    example = {
        "age": 38,
        "workclass": "Private",
        "education": "Bachelors",
        "marital-status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "sex": "Male",
        "hours-per-week": 45,
    }

    result = score_person(example)
    print(f"P(>50K) = {result['probability']:.4f}")
    print(f"defaulted: {', '.join(result['defaulted']) or 'none'}")
    print(f"education-num derived as {result['record']['education-num'].iloc[0]}")

    repeated = score_person(example)["probability"]
    print(f"deterministic: {repeated == result['probability']}")
