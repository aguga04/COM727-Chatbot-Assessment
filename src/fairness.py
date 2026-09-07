"""Audit the deployed classifier for group disparity.

Run from the repository root with ``python -m src.fairness``. Results are
written to ``models/fairness_report.json``, which the application reads, so no
figure is written into the interface by hand.

Four experiments.

1. Group outcome rates. How often the deployed model predicts the upper bracket
   for each group, and how often it is right when it does.
2. Counterfactual flips. Change the sex field on every held-out record and
   rescore. Then repeat, also swapping the household role between husband and
   wife, because the two fields encode the same information and flipping one in
   isolation produces a record the training data never contained.
3. Attribute removal. Retrain with sex, race and country of birth deleted, and
   compare the gap in outcomes against the original.
4. Proxy recovery. Train a classifier to predict sex from the remaining columns,
   first with the household role column present and then with it removed.

Taken together the four measure whether deleting a protected attribute removes
the information it carries. On this dataset it does not.
"""

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src import config
from src.data import load_raw, prepare_features, extract_target
from src.predict import load_artefacts, predict_probabilities
from src.train import MODEL_PARAMS, VALIDATION_SIZE

PROTECTED = ["sex", "race", "native-country"]
DECISION_THRESHOLD = 0.5


def columns_for(attributes, feature_columns):
    """Return the prepared columns produced by the given raw attributes."""
    prefixes = tuple(f"{a}_" for a in attributes)
    return [c for c in feature_columns if c.startswith(prefixes) or c in attributes]


def group_outcomes(df_raw, probabilities, attribute):
    """Return the outcome rates the model produces for each value of an attribute."""
    predicted = (probabilities >= DECISION_THRESHOLD).astype(int)
    actual = extract_target(df_raw).to_numpy()

    rows = []
    for value, index in df_raw.groupby(attribute).groups.items():
        mask = df_raw.index.isin(index)
        positives = actual[mask] == 1
        negatives = ~positives
        rows.append({
            "group": str(value),
            "records": int(mask.sum()),
            "selection_rate": float(predicted[mask].mean()),
            "actual_rate": float(actual[mask].mean()),
            "true_positive_rate": float(predicted[mask][positives].mean())
            if positives.any() else None,
            "false_positive_rate": float(predicted[mask][negatives].mean())
            if negatives.any() else None,
        })

    return sorted(rows, key=lambda r: -r["records"])


def experiment_group_rates(test):
    """How the deployed model's outcomes differ by sex and by race."""
    probabilities = predict_probabilities(test)

    by_sex = group_outcomes(test, probabilities, "sex")
    rates = {r["group"]: r for r in by_sex}

    return {
        "by_sex": by_sex,
        "by_race": group_outcomes(test, probabilities, "race"),
        "selection_rate_gap_sex": float(
            rates["Male"]["selection_rate"] - rates["Female"]["selection_rate"]
        ),
        "true_positive_rate_gap_sex": float(
            rates["Male"]["true_positive_rate"] - rates["Female"]["true_positive_rate"]
        ),
    }


def flip_sex(df_raw, swap_household_role):
    """Return a copy with the sex field reversed on every record.

    With ``swap_household_role`` set, husband and wife are exchanged as well, so
    the record stays internally consistent. Without it, the result includes
    combinations the training data never contained.
    """
    flipped = df_raw.copy()
    flipped["sex"] = flipped["sex"].map({"Male": "Female", "Female": "Male"})

    if swap_household_role:
        flipped["relationship"] = flipped["relationship"].replace(
            {"Husband": "Wife", "Wife": "Husband"}
        )

    return flipped


def experiment_counterfactual(test, swap_household_role):
    """Rescore every held-out record with the sex field reversed."""
    original = predict_probabilities(test)
    flipped = predict_probabilities(flip_sex(test, swap_household_role))

    before = (original >= DECISION_THRESHOLD).astype(int)
    after = (flipped >= DECISION_THRESHOLD).astype(int)
    changed = before != after

    was_male = test["sex"].to_numpy() == "Male"
    lost = changed & was_male & (after == 0)
    gained = changed & ~was_male & (after == 1)

    return {
        "swap_household_role": swap_household_role,
        "records": int(len(test)),
        "predictions_changed": int(changed.sum()),
        "share_changed": float(changed.mean()),
        "men_losing_upper_prediction_when_relabelled_female": int(lost.sum()),
        "women_gaining_upper_prediction_when_relabelled_male": int(gained.sum()),
        "changes_in_other_directions": int(changed.sum() - lost.sum() - gained.sum()),
        "mean_probability_shift": float(np.mean(flipped - original)),
        "mean_absolute_probability_shift": float(np.mean(np.abs(flipped - original))),
    }


def fit_income_model(X_train, y_train, X_val):
    """Fit a model with the deployed hyperparameters and return validation scores."""
    scaler = StandardScaler()
    model = xgb.XGBClassifier(**MODEL_PARAMS)
    model.fit(scaler.fit_transform(X_train), y_train)
    return model.predict_proba(scaler.transform(X_val))[:, 1]


def experiment_removal(train):
    """Retrain without the protected attributes and compare the outcome gap."""
    X, schema = prepare_features(train)
    y = extract_target(train)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VALIDATION_SIZE,
        random_state=config.RANDOM_STATE, stratify=y)

    sex_of_val = train.loc[X_val.index, "sex"].to_numpy()
    protected_columns = columns_for(PROTECTED, schema["feature_columns"])

    results = {}
    for name, frame_train, frame_val in [
        ("with_protected_attributes", X_train, X_val),
        ("without_protected_attributes",
         X_train.drop(columns=protected_columns),
         X_val.drop(columns=protected_columns)),
    ]:
        probabilities = fit_income_model(frame_train, y_train, frame_val)
        predicted = (probabilities >= DECISION_THRESHOLD).astype(int)

        male = sex_of_val == "Male"
        results[name] = {
            "features": int(frame_train.shape[1]),
            "accuracy": float((predicted == y_val.to_numpy()).mean()),
            "selection_rate_male": float(predicted[male].mean()),
            "selection_rate_female": float(predicted[~male].mean()),
            "selection_rate_gap": float(predicted[male].mean() - predicted[~male].mean()),
        }

    results["columns_removed"] = len(protected_columns)
    results["gap_closed"] = float(
        results["with_protected_attributes"]["selection_rate_gap"]
        - results["without_protected_attributes"]["selection_rate_gap"]
    )
    return results


def experiment_proxy_recovery(train):
    """Predict sex from the remaining columns, with and without household role."""
    X, schema = prepare_features(train)
    y = (train["sex"] == "Male").astype(int)

    sex_columns = columns_for(["sex"], schema["feature_columns"])
    role_columns = columns_for(["relationship"], schema["feature_columns"])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VALIDATION_SIZE,
        random_state=config.RANDOM_STATE, stratify=y)

    majority = float(max(y_val.mean(), 1 - y_val.mean()))

    results = {"majority_class_baseline": majority}
    for name, dropped in [
        ("household_role_present", sex_columns),
        ("household_role_removed", sex_columns + role_columns),
    ]:
        probabilities = fit_income_model(
            X_train.drop(columns=dropped), y_train, X_val.drop(columns=dropped))
        predicted = (probabilities >= DECISION_THRESHOLD).astype(int)
        results[name] = {
            "features": int(X_train.shape[1] - len(dropped)),
            "accuracy": float((predicted == y_val.to_numpy()).mean()),
            "lift_over_baseline": float((predicted == y_val.to_numpy()).mean() - majority),
        }

    return results


def run_audit():
    """Run every experiment and write the report."""
    train, test = load_raw()
    metadata = load_artefacts()["metadata"]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_trained_at": metadata["trained_at"],
        "decision_threshold": DECISION_THRESHOLD,
        "held_out_records": int(len(test)),
        "group_rates": experiment_group_rates(test),
        "counterfactual_sex_only": experiment_counterfactual(test, False),
        "counterfactual_sex_and_role": experiment_counterfactual(test, True),
        "attribute_removal": experiment_removal(train),
        "proxy_recovery": experiment_proxy_recovery(train),
    }

    with open(config.MODELS / "fairness_report.json", "w") as handle:
        json.dump(report, handle, indent=2)

    return report


def report_summary(report):
    """Print the findings in the order the report should make the argument."""
    rates = report["group_rates"]
    print("Outcome rates on the held-out file")
    print(pd.DataFrame(rates["by_sex"]).to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"selection rate gap {rates['selection_rate_gap_sex']:+.4f}   "
          f"true positive rate gap {rates['true_positive_rate_gap_sex']:+.4f}")

    for key, title in [("counterfactual_sex_only", "sex flipped alone"),
                       ("counterfactual_sex_and_role", "sex and household role flipped")]:
        c = report[key]
        print(f"\nCounterfactual, {title}")
        print(f"  predictions changed          {c['predictions_changed']} "
              f"of {c['records']} ({c['share_changed']:.2%})")
        print(f"  men losing upper prediction  {c['men_losing_upper_prediction_when_relabelled_female']}")
        print(f"  women gaining it             {c['women_gaining_upper_prediction_when_relabelled_male']}")
        print(f"  changes in other directions  {c['changes_in_other_directions']}")
        print(f"  mean probability shift       {c['mean_probability_shift']:+.4f}")

    r = report["attribute_removal"]
    print(f"\nAttribute removal ({r['columns_removed']} prepared columns deleted)")
    for key in ["with_protected_attributes", "without_protected_attributes"]:
        v = r[key]
        print(f"  {key:<30} accuracy {v['accuracy']:.4f}   "
              f"male {v['selection_rate_male']:.4f}   female {v['selection_rate_female']:.4f}   "
              f"gap {v['selection_rate_gap']:+.4f}")
    print(f"  gap closed by removal          {r['gap_closed']:+.4f}")

    p = report["proxy_recovery"]
    print(f"\nProxy recovery (majority class baseline {p['majority_class_baseline']:.4f})")
    for key in ["household_role_present", "household_role_removed"]:
        v = p[key]
        print(f"  {key:<25} accuracy {v['accuracy']:.4f}   "
              f"lift {v['lift_over_baseline']:+.4f}")


if __name__ == "__main__":
    report_summary(run_audit())
    print(f"\nWritten to {config.MODELS / 'fairness_report.json'}")
