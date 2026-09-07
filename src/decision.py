"""Map a predicted probability onto a decision band.

The application does not return a bare yes or no. It routes a case into one of
three bands: confidently below the census threshold, confidently above it, or an
uncertain middle band that is escalated to a human assessor rather than decided
automatically. This is the accept, refer, decline pattern used in credit and
lending screening.

Boundaries are read from ``models/decision_thresholds.json``, written by the
notebook when it selects them. They are not literals in this file, so a change
of policy is a rerun of that analysis rather than an edit to code.

The wording returned here does not state what a person earns. It states which
side of the threshold the model places them on.
"""

import json

from src import config

THRESHOLD_LABEL = "the 1994 census threshold of fifty thousand US dollars"

BANDS = {
    "below": {
        "label": "Likely below the threshold",
        "action": "Proceed on the basis that this person falls in the lower bracket.",
        "interpretation": (
            "The model is reasonably confident this person falls below "
            f"{THRESHOLD_LABEL}."
        ),
    },
    "referral": {
        "label": "Refer for human assessment",
        "action": "Do not decide automatically. Escalate this case to an assessor.",
        "interpretation": (
            "The model cannot separate this case. Records in this band split "
            "almost evenly between the two brackets, so the prediction carries "
            "little information and should not be acted on without a human "
            "reviewing it."
        ),
    },
    "above": {
        "label": "Likely above the threshold",
        "action": "Proceed on the basis that this person falls in the upper bracket.",
        "interpretation": (
            "The model is reasonably confident this person falls above "
            f"{THRESHOLD_LABEL}."
        ),
    },
}

_thresholds = None


def load_thresholds():
    """Return the boundaries and the evidence recorded when they were chosen."""
    global _thresholds

    if _thresholds is None:
        with open(config.MODELS / "decision_thresholds.json") as handle:
            _thresholds = json.load(handle)

    return _thresholds


def band_of(probability):
    """Return the band key for a probability.

    Boundaries follow the analysis: below the lower bound is the lower band,
    above the upper bound is the upper band, and the boundaries themselves fall
    inside the referral band.
    """
    thresholds = load_thresholds()

    if probability < thresholds["lower"]:
        return "below"
    if probability > thresholds["upper"]:
        return "above"
    return "referral"


def band_range(band):
    """Return the probability range of a band as displayable text."""
    thresholds = load_thresholds()
    low, high = thresholds["lower"], thresholds["upper"]

    if band == "below":
        return f"p < {low:g}"
    if band == "above":
        return f"p > {high:g}"
    return f"{low:g} to {high:g}"


def decide(probability):
    """Turn a probability into a band, an interpretation and a recommended action."""
    band = band_of(probability)

    return {
        "probability": float(probability),
        "band": band,
        "label": BANDS[band]["label"],
        "range": band_range(band),
        "action": BANDS[band]["action"],
        "interpretation": BANDS[band]["interpretation"],
        "is_referral": band == "referral",
    }


def band_evidence():
    """Return the figures supporting the boundaries.

    Every value comes from the validation analysis, so a rerun of the notebook
    updates what the application reports.
    """
    thresholds = load_thresholds()

    return {
        "lower": thresholds["lower"],
        "upper": thresholds["upper"],
        "referral_rate": thresholds["referral_rate"],
        "automated_accuracy": thresholds["automated_accuracy"],
        "baseline_accuracy": thresholds["baseline_accuracy_single_threshold"],
        "referred_accuracy": thresholds["referred_accuracy"],
        "accuracy_gain": thresholds["accuracy_gain"],
        "validation_records": thresholds["validation_records"],
        "brier_score": thresholds["calibration"]["brier_score"],
        "expected_calibration_error": thresholds["calibration"][
            "expected_calibration_error"
        ],
    }


def evidence_sentence():
    """Return a one paragraph justification of the bands, composed from the figures."""
    e = band_evidence()

    return (
        f"The boundaries of {e['lower']:g} and {e['upper']:g} were selected on "
        f"{e['validation_records']:,} validation records. They send "
        f"{e['referral_rate']:.1%} of cases to a human assessor and raise accuracy "
        f"on the remaining automated decisions from {e['baseline_accuracy']:.1%} to "
        f"{e['automated_accuracy']:.1%}. Inside the referral band the model is only "
        f"{e['referred_accuracy']:.1%} accurate, which is close to chance and is the "
        f"evidence that these are cases it genuinely cannot separate."
    )


if __name__ == "__main__":
    print(evidence_sentence())
    print()

    for probability in [0.02, 0.29, 0.30, 0.50, 0.70, 0.71, 0.95]:
        outcome = decide(probability)
        flag = "REFER" if outcome["is_referral"] else "     "
        print(f"{probability:>5.2f}  {flag}  {outcome['label']:<28} ({outcome['range']})")

    print()
    print(decide(0.4763)["interpretation"])
