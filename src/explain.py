"""Decompose a prediction into per attribute contributions.

A gradient boosted ensemble has no coefficients, so contributions cannot be
obtained by multiplying feature values by weights. XGBoost implements TreeSHAP
internally and exposes it through the prediction contributions option on the
underlying booster. For one record it returns a value per feature plus a bias
term, and those values sum to the raw margin. For a binary logistic objective
the raw margin is the log odds, so applying the sigmoid to the sum reproduces
the predicted probability exactly.

Contributions are expressed in log odds. They are not percentages and not
amounts of money: a contribution of 0.8 adds 0.8 to the log odds. Any interface
displaying these values labels the unit.

One hot encoding splits a single attribute across several columns, only one of
which is active for a given record. The attribution is additive, so those
columns are summed to give one contribution per raw attribute.
"""

import numpy as np
import pandas as pd
import xgboost as xgb

from src import config
from src.predict import load_artefacts


def source_attribute(column):
    """Return the raw attribute a prepared feature column came from.

    Numeric columns carry the attribute name. Dummy columns are named
    ``attribute_value``, so the prefix is matched against the known categorical
    names.
    """
    for attribute in config.CATEGORICAL_COLS:
        if column.startswith(f"{attribute}_"):
            return attribute
    return column


def column_contributions(scaled, feature_names):
    """Return the log odds contribution of every prepared feature column.

    The final element of the booster output is the bias term, returned
    separately.
    """
    booster = load_artefacts()["booster"]
    matrix = xgb.DMatrix(scaled)

    raw = booster.predict(matrix, pred_contribs=True)[0]
    contributions = pd.Series(raw[:-1], index=feature_names)
    bias = float(raw[-1])

    return contributions, bias


def describe_values(record):
    """Return the value each raw attribute took, as displayable text."""
    row = record.iloc[0]
    values = {}
    for attribute in config.NUMERIC_COLS + config.CATEGORICAL_COLS:
        if attribute in row:
            values[attribute] = row[attribute]
    return values


def explain(result):
    """Decompose a scored record into contributions grouped by attribute.

    ``result`` is the dictionary returned by ``src.predict.score_person``.

    Returns the attribute table sorted by absolute contribution, the bias term,
    the reconstructed log odds, the probability, and the difference between that
    probability and the one the model reported.
    """
    contributions, bias = column_contributions(result["scaled"], result["feature_names"])

    attributes = contributions.groupby(source_attribute).sum()
    values = describe_values(result["record"])

    table = pd.DataFrame({
        "attribute": attributes.index,
        "value": [values.get(a) for a in attributes.index],
        "contribution": attributes.values,
    })
    table["direction"] = np.where(table["contribution"] >= 0, "towards", "away")
    table = table.reindex(
        table["contribution"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)

    log_odds = float(contributions.sum() + bias)
    probability = 1.0 / (1.0 + np.exp(-log_odds))

    return {
        "table": table,
        "bias": bias,
        "log_odds": log_odds,
        "probability": float(probability),
        "reconstruction_error": abs(probability - result["probability"]),
    }


def top_factor(explanation, direction=None):
    """Return the single largest contribution, optionally in one direction."""
    table = explanation["table"]

    if direction == "towards":
        table = table[table["contribution"] > 0]
    elif direction == "away":
        table = table[table["contribution"] < 0]

    if table.empty:
        return None
    return table.iloc[0].to_dict()


if __name__ == "__main__":
    from src.predict import score_person

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
    explanation = explain(result)

    print(explanation["table"].to_string(index=False,
                                         float_format=lambda v: f"{v: .4f}"))
    print()
    print(f"bias            {explanation['bias']: .6f}")
    print(f"contributions   {explanation['table']['contribution'].sum(): .6f}")
    print(f"log odds        {explanation['log_odds']: .6f}")
    print(f"probability      {explanation['probability']:.6f}")
    print(f"model output     {result['probability']:.6f}")
    print(f"reconstruction error {explanation['reconstruction_error']:.2e}")
    print()
    print("largest factor towards:", top_factor(explanation, "towards")["attribute"])
    print("largest factor away:   ", top_factor(explanation, "away")["attribute"])
