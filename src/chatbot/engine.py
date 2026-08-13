"""Route a question to an intent and compose the answer.

Two steps. The intent classifier decides what was asked; the response engine
decides what to say. Answers about the current assessment are assembled from
live model output rather than selected from stored sentences, so changing the
person changes the wording and the figures. Static answers are used only where
nothing depends on the person, such as a greeting or the description of the
dataset.

No figure is written into this file or into ``data/intents.json``. Every number
is read from the artefacts at the moment it is spoken.
"""

import json
import random
from functools import lru_cache

import joblib
import numpy as np

from src import config
from src.chatbot import limitations
from src.decision import decide, band_evidence, evidence_sentence
from src.explain import explain, top_factor
from src.nlp import vectorise
from src.predict import load_artefacts
from src.train import load_intents

READABLE = {
    "age": "age",
    "education": "highest qualification",
    "education-num": "years of education",
    "capital-gain": "capital gains",
    "capital-loss": "capital losses",
    "hours-per-week": "hours worked per week",
    "workclass": "employer type",
    "marital-status": "marital status",
    "occupation": "occupation",
    "relationship": "household role",
    "race": "race",
    "sex": "sex",
    "native-country": "country of birth",
}

NEEDS_ASSESSMENT = (
    "I need someone to assess first. Fill in the details on the assessment tab "
    "and I can answer that."
)

FALLBACK = (
    "I am not confident I understood that. I can answer questions about the "
    "current assessment, the algorithm behind it, the census data it learned "
    "from, and the limitations of the whole system."
)

_rng = random.Random()


@lru_cache(maxsize=1)
def load_classifier():
    """Return the trained intent classifier and its vocabulary."""
    return joblib.load(config.MODELS / "intent_model.joblib")


@lru_cache(maxsize=1)
def confidence_threshold():
    """Return the fallback threshold chosen in the notebook evaluation."""
    with open(config.MODELS / "intent_metrics.json") as handle:
        return float(json.load(handle)["confidence_threshold"])


@lru_cache(maxsize=1)
def intents_by_tag():
    """Return the intent definitions keyed by tag."""
    return {intent["tag"]: intent for intent in load_intents()}


def classify(text):
    """Return the recognised intent, its confidence, and whether it clears the threshold."""
    artefact = load_classifier()

    vector = vectorise(text, artefact["vocabulary"], binary=artefact["binary"])
    probabilities = artefact["model"].predict_proba(vector.reshape(1, -1))[0]

    position = int(probabilities.argmax())
    confidence = float(probabilities[position])

    return {
        "tag": artefact["tags"][position],
        "confidence": confidence,
        "threshold": confidence_threshold(),
        "accepted": confidence >= confidence_threshold(),
    }


@lru_cache(maxsize=1)
def static_context():
    """Return every value that does not depend on the person being assessed."""
    metadata = load_artefacts()["metadata"]
    held_out = metadata["metrics"]["held_out_test"]
    evidence = band_evidence()

    with open(config.MODELS / "decision_thresholds.json") as handle:
        thresholds = json.load(handle)

    if "band_actual_rates" not in thresholds:
        raise KeyError(
            "decision_thresholds.json has no band_actual_rates entry. Rerun the "
            "export cell at the end of Section 6 of the notebook, which adds it."
        )

    return {
        "model_name": "XGBoost",
        "train_rows": f"{metadata['rows']['train']:,}",
        "test_rows": f"{metadata['rows']['held_out_test']:,}",
        "features": metadata["feature_count"],
        "accuracy_pct": f"{held_out['accuracy']:.1%}",
        "precision_pct": f"{held_out['precision']:.1%}",
        "recall_pct": f"{held_out['recall']:.1%}",
        "f1": f"{held_out['f1']:.3f}",
        "roc_auc": f"{held_out['roc_auc']:.3f}",
        "lower": f"{evidence['lower']:g}",
        "upper": f"{evidence['upper']:g}",
        "referral_rate_pct": f"{evidence['referral_rate']:.1%}",
        "referred_accuracy_pct": f"{evidence['referred_accuracy']:.1%}",
        "referred_actual_pct": f"{thresholds['band_actual_rates']['referred']:.1%}",
        "brier": f"{evidence['brier_score']:.4f}",
        "ece": f"{evidence['expected_calibration_error']:.4f}",
        "threshold_evidence": evidence_sentence(),
        "limitations_summary": limitations.summary_paragraph(),
    }


def format_contribution(value):
    """Return a signed log odds figure as displayable text."""
    return f"{value:+.3f} in log odds"


def format_probability(value):
    """Return a probability as text, without collapsing small values to zero.

    Straight percentage rounding turns 0.0004 into 0.0 percent, which reads as
    an impossibility rather than as a small number.
    """
    if value < 0.001:
        return "under 0.1%"
    if value > 0.999:
        return "over 99.9%"
    return f"{value:.1%}"


def format_value(value):
    """Return an attribute value as displayable text."""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return str(int(value)) if float(value).is_integer() else f"{value:.2f}"
    return str(value)


def person_context(assessment):
    """Return the values that describe the person currently assessed.

    ``assessment`` is the dictionary returned by ``src.predict.score_person``.
    """
    explanation = explain(assessment)
    outcome = decide(assessment["probability"])

    toward = top_factor(explanation, "towards")
    away = top_factor(explanation, "away")

    lines = []
    for row in explanation["table"].itertuples():
        name = READABLE.get(row.attribute, row.attribute)
        lines.append(f"  {name}: {format_value(row.value)}  "
                     f"{format_contribution(row.contribution)}")

    context = {
        "probability_pct": format_probability(assessment["probability"]),
        "band_label": outcome["label"].lower(),
        "band_range": outcome["range"],
        "band_action": outcome["action"],
        "band_interpretation": outcome["interpretation"],
        "breakdown": "\n".join(lines),
        "defaulted_list": ", ".join(
            READABLE.get(f, f) for f in assessment["defaulted"]) or "none",
        "top_toward_attr": READABLE.get(toward["attribute"], toward["attribute"]) if toward else "nothing",
        "top_toward_value": format_value(toward["value"]) if toward else "",
        "top_toward_contrib": format_contribution(toward["contribution"]) if toward else "",
        "top_away_attr": READABLE.get(away["attribute"], away["attribute"]) if away else "nothing",
        "top_away_value": format_value(away["value"]) if away else "",
        "top_away_contrib": format_contribution(away["contribution"]) if away else "",
    }

    context["_explanation"] = explanation
    return context


def attribute_context(explanation, attribute):
    """Return the contribution of one named attribute."""
    table = explanation["table"]
    match = table[table["attribute"] == attribute]

    if match.empty:
        return {"attr_name": READABLE.get(attribute, attribute),
                "attr_value": "not recorded",
                "attr_contribution": "nothing",
                "attr_direction": "neither towards nor away from"}

    row = match.iloc[0]
    return {
        "attr_name": READABLE.get(attribute, attribute),
        "attr_value": format_value(row["value"]),
        "attr_contribution": format_contribution(row["contribution"]),
        "attr_direction": "towards" if row["contribution"] >= 0 else "away from",
    }


def build_context(intent, assessment):
    """Assemble every value the chosen intent's responses might refer to."""
    context = dict(static_context())

    if intent.get("section"):
        context["limitation_section"] = limitations.section(intent["section"])["body"]

    if assessment is not None:
        person = person_context(assessment)
        explanation = person.pop("_explanation")
        context.update(person)

        if intent.get("attribute"):
            context.update(attribute_context(explanation, intent["attribute"]))

    return context


def respond(text, assessment=None, rng=None):
    """Classify a message and compose the answer.

    Returns the recognised intent, the confidence behind it, and the response.
    Questions that describe a person return a prompt instead when nothing has
    been assessed yet.
    """
    recognised = classify(text)

    if not recognised["accepted"]:
        return {**recognised, "tag": "fallback", "response": FALLBACK, "used_fallback": True}

    intent = intents_by_tag()[recognised["tag"]]

    if intent["show_when"] == "after_assessment" and assessment is None:
        return {**recognised, "response": NEEDS_ASSESSMENT, "used_fallback": False}

    context = build_context(intent, assessment)
    template = (rng or _rng).choice(intent["responses"])

    return {**recognised, "response": template.format(**context), "used_fallback": False}


def question_buttons(assessment_ready):
    """Return the questions to offer, grouped by category and filtered by state."""
    grouped = {}

    for intent in load_intents():
        if intent["show_when"] == "after_assessment" and not assessment_ready:
            continue
        grouped.setdefault(intent["category"], []).append(
            {"tag": intent["tag"], "display": intent["display"]})

    return grouped


if __name__ == "__main__":
    from src.predict import score_person

    person = score_person({
        "age": 38, "workclass": "Private", "education": "Bachelors",
        "marital-status": "Married-civ-spouse", "occupation": "Prof-specialty",
        "relationship": "Husband", "sex": "Male", "hours-per-week": 45,
    })

    rng = random.Random(0)
    failures = []

    for intent in load_intents():
        outcome = respond(intent["display"], assessment=person, rng=rng)
        if outcome["tag"] != intent["tag"]:
            failures.append((intent["display"], intent["tag"], outcome["tag"]))
        preview = outcome["response"].replace("\n", " ")[:96]
        print(f"[{outcome['confidence']:.3f}] {intent['tag']:<24} {preview}")

    print(f"\nintents answered: {len(load_intents())}   misrouted: {len(failures)}")
    for display, expected, got in failures:
        print(f"  '{display}' -> {got} instead of {expected}")

    print("\nWithout an assessment loaded:")
    print(" ", respond("What did the model decide?")["response"])
    print("\nUnrecognised input:")
    unknown = respond("purple monkey dishwasher")
    print(f"  [{unknown['confidence']:.3f}] {unknown['response'][:80]}")