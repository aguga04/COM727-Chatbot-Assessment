"""Automated checks for the parts of the pipeline that fail quietly.

Run from the repository root with ``pytest -q``.

Most of these guard against defects that produce a plausible wrong answer
rather than an error, which is the class of problem that survives manual
testing. Two in particular are regression tests for defects found during
development: the single record encoding defect, and a display string leaking
into its own training patterns.

The suite assumes the artefacts exist. Run ``python -m src.train`` and
``python -m src.fairness`` first.
"""

import json

import numpy as np
import pytest

from src import config
from src.data import load_raw, prepare_features, extract_target, compute_defaults
from src.decision import band_of, decide, load_thresholds
from src.explain import explain
from src.chatbot.engine import classify, respond, question_buttons
from src.nlp import preprocess, vectorise
from src.predict import load_artefacts, predict_probabilities, score_person
from src.train import load_intents, training_examples

BASE_PERSON = {
    "age": 38,
    "workclass": "Private",
    "education": "Bachelors",
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "sex": "Male",
    "hours-per-week": 45,
}


@pytest.fixture(scope="session")
def raw():
    return load_raw()


@pytest.fixture(scope="session")
def artefacts():
    return load_artefacts()


@pytest.fixture(scope="session")
def assessment():
    return score_person(BASE_PERSON)


@pytest.fixture(scope="session")
def intents():
    return load_intents()


# --------------------------------------------------------------------- data

def test_raw_files_load_with_the_published_shapes(raw):
    train, test = raw
    assert train.shape == (32561, 15)
    assert test.shape == (16281, 15)


def test_test_labels_have_no_trailing_full_stop(raw):
    _, test = raw
    assert set(test[config.TARGET].unique()) == {"<=50K", ">50K"}


def test_target_is_binary_and_matches_the_published_class_balance(raw):
    train, _ = raw
    target = extract_target(train)
    assert set(target.unique()) == {0, 1}
    assert target.mean() == pytest.approx(0.2408, abs=1e-4)


def test_a_single_record_produces_the_training_columns_in_training_order(artefacts):
    schema = artefacts["schema"]
    person, _ = _one_row(artefacts)
    features, _ = prepare_features(person, schema)

    assert list(features.columns) == schema["feature_columns"]
    assert features.shape == (1, len(schema["feature_columns"]))


def test_changing_one_categorical_field_changes_the_feature_vector(artefacts):
    """Regression test for the single record encoding defect.

    Dummy encoding with the first level dropped silently collapses a lone
    record's categorical column, so two people differing only in sex produced
    identical vectors and the model read both as the reference category. The
    failure raised nothing, which is why it is tested rather than trusted.
    """
    schema = artefacts["schema"]

    for field, first, second in [("sex", "Male", "Female"),
                                 ("relationship", "Husband", "Wife"),
                                 ("workclass", "Private", "Self-emp-not-inc")]:
        one, _ = _one_row(artefacts, {field: first})
        two, _ = _one_row(artefacts, {field: second})

        vector_one, _ = prepare_features(one, schema)
        vector_two, _ = prepare_features(two, schema)

        assert not np.array_equal(vector_one.values, vector_two.values), (
            f"records differing only in {field} produced identical features")


def test_an_unseen_category_falls_back_rather_than_failing(artefacts):
    person, _ = _one_row(artefacts, {"native-country": "Atlantis"})
    features, _ = prepare_features(person, artefacts["schema"])

    assert features.shape[1] == len(artefacts["schema"]["feature_columns"])
    assert features["native-country_Other"].iloc[0] == 1.0


def test_defaults_cover_every_raw_attribute(raw):
    train, _ = raw
    defaults = compute_defaults(train)
    expected = set(config.NUMERIC_COLS) | set(config.CATEGORICAL_COLS)
    assert set(defaults) == expected


def _one_row(artefacts, overrides=None):
    from src.data import build_person
    answers = dict(BASE_PERSON)
    answers.update(overrides or {})
    return build_person(answers, artefacts["defaults"])


# ---------------------------------------------------------------- inference

def test_scoring_is_deterministic():
    first = score_person(BASE_PERSON)["probability"]
    second = score_person(BASE_PERSON)["probability"]
    assert first == second


def test_single_record_scoring_agrees_with_batch_scoring(raw):
    """The batch path is known correct, so the single record path is checked against it."""
    _, test = raw
    sample = test.head(40)

    batch = predict_probabilities(sample)
    singles = [
        score_person({k: v for k, v in row.items() if k != config.TARGET})["probability"]
        for _, row in sample.iterrows()
    ]

    assert np.allclose(batch, singles, atol=1e-6)


def test_years_of_education_are_derived_rather_than_defaulted(assessment):
    assert "education-num" not in assessment["defaulted"]
    assert assessment["record"]["education-num"].iloc[0] == 13


def test_held_out_accuracy_matches_the_recorded_metadata(raw, artefacts):
    _, test = raw
    probabilities = predict_probabilities(test)
    accuracy = ((probabilities >= 0.5).astype(int) == extract_target(test).to_numpy()).mean()
    recorded = artefacts["metadata"]["metrics"]["held_out_test"]["accuracy"]
    assert accuracy == pytest.approx(recorded, abs=1e-9)


# -------------------------------------------------------------- explanation

def test_contributions_reconstruct_the_model_output(assessment):
    explanation = explain(assessment)
    total = explanation["table"]["contribution"].sum()
    log_odds = total + explanation["bias"]

    assert log_odds == pytest.approx(explanation["log_odds"], abs=1e-5)
    assert 1 / (1 + np.exp(-log_odds)) == pytest.approx(assessment["probability"], abs=1e-6)
    assert explanation["reconstruction_error"] < 1e-5


def test_grouping_dummy_columns_loses_nothing(assessment):
    from src.explain import column_contributions

    columns, _ = column_contributions(assessment["scaled"], assessment["feature_names"])
    grouped = explain(assessment)["table"]["contribution"].sum()

    assert grouped == pytest.approx(columns.sum(), abs=1e-5)


def test_one_row_per_raw_attribute(assessment):
    table = explain(assessment)["table"]
    expected = set(config.NUMERIC_COLS) | set(config.CATEGORICAL_COLS)
    assert set(table["attribute"]) == expected


# ------------------------------------------------------------------- bands

def test_every_boundary_maps_to_the_intended_band():
    thresholds = load_thresholds()
    low, high = thresholds["lower"], thresholds["upper"]

    assert band_of(low - 1e-9) == "below"
    assert band_of(low) == "referral"
    assert band_of((low + high) / 2) == "referral"
    assert band_of(high) == "referral"
    assert band_of(high + 1e-9) == "above"


def test_the_referral_band_returns_an_escalation_not_a_verdict():
    thresholds = load_thresholds()
    middle = decide((thresholds["lower"] + thresholds["upper"]) / 2)

    assert middle["is_referral"]
    assert "escalate" in middle["action"].lower()


def test_no_band_wording_claims_an_income_figure():
    for probability in [0.05, 0.5, 0.95]:
        outcome = decide(probability)
        text = f"{outcome['label']} {outcome['action']} {outcome['interpretation']}"
        assert "earns" not in text.lower()
        assert "salary" not in text.lower()


# ------------------------------------------------------- conversational data

def test_the_intents_file_meets_the_brief(intents):
    assert 30 <= len(intents) <= 40

    for intent in intents:
        assert len(intent["patterns"]) >= 3, intent["tag"]
        assert len(intent["responses"]) >= 3, intent["tag"]
        assert intent["show_when"] in {"always", "after_assessment"}


def test_intent_tags_are_unique(intents):
    tags = [intent["tag"] for intent in intents]
    assert len(set(tags)) == len(tags)


def test_no_display_string_appears_among_its_own_patterns(intents):
    """Regression test. A memorised button label makes the button test a tautology."""
    for intent in intents:
        display = intent["display"].lower().strip("?. ")
        patterns = [p.lower().strip("?. ") for p in intent["patterns"]]
        assert display not in patterns, intent["tag"]


def test_the_training_set_builder_rejects_a_leaked_display_string(intents):
    leaked = [dict(intent) for intent in intents]
    leaked[0]["patterns"] = leaked[0]["patterns"] + [leaked[0]["display"]]

    with pytest.raises(ValueError):
        training_examples(leaked)


def test_no_response_states_a_stored_metric(intents):
    """Results belong in the generated files, not in the authored responses.

    The target is a metric that would go stale on retraining, which in this
    project always arrives as a percentage, a figure to three or more decimal
    places, or a thousands separated count. Round illustrative numbers such as
    the 0.8 used to explain what a log odds contribution is not are left alone.
    """
    import re

    metric = re.compile(r"\b\d+(\.\d{3,})?%|\b\d+\.\d{3,}\b|\b\d{1,3}(,\d{3})+\b")

    for intent in intents:
        for response in intent["responses"]:
            found = metric.search(response)
            assert found is None, f"{intent['tag']} states {found.group()}"


# ------------------------------------------------------- text preprocessing

def test_preprocessing_is_deterministic():
    artefacts = load_artefacts()
    vocabulary = sorted(set(preprocess("how much did age matter")))

    first = vectorise("how much did age matter", vocabulary)
    second = vectorise("how much did age matter", vocabulary)
    assert np.array_equal(first, second)
    assert artefacts is not None


def test_stopwords_are_removed_but_interrogatives_survive():
    assert preprocess("the a of to and") == []
    assert "what" in preprocess("what is this")
    assert "how" in preprocess("how does it work")


def test_inflected_forms_reduce_to_a_shared_base():
    """Verb before noun, or does becomes doe and was becomes wa."""
    assert preprocess("does")[0] == preprocess("did")[0]
    assert preprocess("is")[0] == preprocess("was")[0]
    assert preprocess("records")[0] == "record"


def test_different_phrasings_of_one_question_share_content_words():
    variants = ["how much did age matter", "did age matter for this person",
                "does their age carry weight"]
    shared = set.intersection(*[set(preprocess(v)) for v in variants])
    assert "age" in shared


# ----------------------------------------------------------- intent routing

def test_every_display_string_reaches_its_own_intent_above_the_threshold(intents):
    """The figure that determines whether the deployed interface works at all."""
    for intent in intents:
        outcome = classify(intent["display"])
        assert outcome["tag"] == intent["tag"], intent["display"]
        assert outcome["accepted"], (
            f"{intent['display']} scored {outcome['confidence']:.3f}, below threshold")


def test_every_intent_composes_a_response_without_a_missing_placeholder(intents, assessment):
    for intent in intents:
        outcome = respond(intent["display"], assessment=assessment)
        assert outcome["tag"] == intent["tag"]
        assert outcome["response"].strip()
        assert "{" not in outcome["response"]


def test_unrecognised_input_reaches_the_fallback():
    outcome = respond("purple monkey dishwasher")
    assert outcome["used_fallback"]
    assert outcome["confidence"] < outcome["threshold"]


def test_answers_about_a_person_change_when_the_person_changes():
    import random

    first = score_person(BASE_PERSON)
    second = score_person({**BASE_PERSON, "age": 22, "education": "HS-grad",
                           "occupation": "Other-service", "hours-per-week": 18})

    for question in ["Which factor mattered most?", "How likely is the upper bracket?"]:
        one = respond(question, first, rng=random.Random(0))["response"]
        two = respond(question, second, rng=random.Random(0))["response"]
        assert one != two, question


def test_static_answers_do_not_change_with_the_person():
    import random

    first = score_person(BASE_PERSON)
    second = score_person({**BASE_PERSON, "age": 22})

    one = respond("How accurate is it?", first, rng=random.Random(0))["response"]
    two = respond("How accurate is it?", second, rng=random.Random(0))["response"]
    assert one == two


def test_prediction_questions_are_withheld_until_someone_is_assessed():
    before = question_buttons(False)
    after = question_buttons(True)

    assert sum(len(v) for v in before.values()) < sum(len(v) for v in after.values())
    assert "This prediction" in after


# ------------------------------------------------------------- generated files

def test_every_generated_file_the_interface_reads_exists():
    required = [
        "income_model.json", "preprocessing.joblib", "metadata.json",
        "intent_model.joblib", "intent_metrics.json",
        "decision_thresholds.json", "fairness_report.json",
        "model_comparison.csv", "val_vs_test_all_models.csv",
    ]
    missing = [name for name in required if not (config.MODELS / name).exists()]
    assert not missing, f"missing artefacts: {missing}"


def test_the_thresholds_file_carries_the_band_outcome_rates():
    thresholds = load_thresholds()
    assert "band_actual_rates" in thresholds
    assert set(thresholds["band_actual_rates"]) == {"below", "referred", "above"}


def test_recorded_library_versions_match_the_versions_in_use():
    from src.predict import version_report
    assert version_report() == []


def test_limitations_text_contains_no_unfilled_placeholder():
    from src.chatbot import limitations

    for section in limitations.panel_text():
        assert "{" not in section["body"], section["key"]
        assert "}" not in section["body"], section["key"]