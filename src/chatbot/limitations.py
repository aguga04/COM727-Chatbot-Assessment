"""Standing limitations content for the interface and the conversational layer.

Every figure in this text is read from the generated files rather than written
into it. Retraining the model or rerunning the audit changes what the
application says, so the interface can never state a number the artefacts no
longer support.

The text is deliberately written to be lifted into the ethics section of the
report. It is not a disclaimer bolted on at the end; the aim statement commits
the project to describing applications and limitations, so this is a stated
design feature.
"""

import json

from src import config
from src.predict import load_artefacts

_figures = None


def figures():
    """Collect every number the limitations text refers to, from the artefacts."""
    global _figures

    if _figures is not None:
        return _figures

    metadata = load_artefacts()["metadata"]

    with open(config.MODELS / "fairness_report.json") as handle:
        audit = json.load(handle)

    with open(config.MODELS / "decision_thresholds.json") as handle:
        thresholds = json.load(handle)

    rates = {r["group"]: r for r in audit["group_rates"]["by_sex"]}
    flip = audit["counterfactual_sex_only"]
    removal = audit["attribute_removal"]
    proxy = audit["proxy_recovery"]

    _figures = {
        "trained_at": metadata["trained_at"][:10],
        "train_rows": f"{metadata['rows']['train']:,}",
        "test_rows": f"{metadata['rows']['held_out_test']:,}",
        "features": metadata["feature_count"],
        "accuracy": f"{metadata['metrics']['held_out_test']['accuracy']:.1%}",
        "recall": f"{metadata['metrics']['held_out_test']['recall']:.1%}",
        "lower": f"{thresholds['lower']:g}",
        "upper": f"{thresholds['upper']:g}",
        "referral_rate": f"{thresholds['referral_rate']:.1%}",
        "automated_accuracy": f"{thresholds['automated_accuracy']:.1%}",
        "referred_accuracy": f"{thresholds['referred_accuracy']:.1%}",
        "referred_actual_rate": f"{thresholds['band_actual_rates']['referred']:.1%}",
        "male_selection": f"{rates['Male']['selection_rate']:.1%}",
        "female_selection": f"{rates['Female']['selection_rate']:.1%}",
        "male_actual": f"{rates['Male']['actual_rate']:.1%}",
        "female_actual": f"{rates['Female']['actual_rate']:.1%}",
        "male_tpr": f"{rates['Male']['true_positive_rate']:.1%}",
        "female_tpr": f"{rates['Female']['true_positive_rate']:.1%}",
        "flipped": f"{flip['predictions_changed']:,}",
        "flipped_share": f"{flip['share_changed']:.2%}",
        "men_losing": f"{flip['men_losing_upper_prediction_when_relabelled_female']:,}",
        "women_gaining": f"{flip['women_gaining_upper_prediction_when_relabelled_male']:,}",
        "other_direction": flip["changes_in_other_directions"],
        "gap_with": f"{removal['with_protected_attributes']['selection_rate_gap']:.3f}",
        "gap_without": f"{removal['without_protected_attributes']['selection_rate_gap']:.3f}",
        "gap_share_closed": f"{removal['gap_closed'] / removal['with_protected_attributes']['selection_rate_gap']:.0%}",
        "columns_removed": removal["columns_removed"],
        "accuracy_with": f"{removal['with_protected_attributes']['accuracy']:.4f}",
        "accuracy_without": f"{removal['without_protected_attributes']['accuracy']:.4f}",
        "proxy_with_role": f"{proxy['household_role_present']['accuracy']:.1%}",
        "proxy_without_role": f"{proxy['household_role_removed']['accuracy']:.1%}",
        "proxy_baseline": f"{proxy['majority_class_baseline']:.1%}",
    }

    return _figures


SECTIONS = [
    {
        "key": "dataset_age",
        "heading": "The data is from 1994",
        "template": (
            "This model learned from the UCI Adult dataset, drawn from the 1994 United States "
            "Census. The target it predicts is whether a person's annual income exceeded fifty "
            "thousand dollars, a threshold fixed in 1994 and never adjusted since. Three decades "
            "of inflation mean that boundary no longer marks the same position in the income "
            "distribution that it did when the data was collected, so a prediction of the upper "
            "bracket does not correspond to any meaningful income level today. Ding and "
            "colleagues document this problem in detail and argue the dataset should be retired "
            "from fairness research for exactly this reason."
        ),
    },
    {
        "key": "geography",
        "heading": "The data describes the United States",
        "template": (
            "Every record comes from the US Census. Occupational categories, education levels, "
            "employment classes and the relationship between them are specific to the United "
            "States in the mid nineteen nineties. Nothing here transfers to another country's "
            "labour market, and the fifty thousand dollar figure is not converted to any other "
            "currency."
        ),
    },
    {
        "key": "not_income",
        "heading": "A band is not an income estimate",
        "template": (
            "The model outputs a probability that a person falls above the census threshold. It "
            "does not estimate what anyone earns, and the dataset contains no continuous income "
            "figure from which such an estimate could be made. The three bands are confidence "
            "bands about which side of the threshold a person falls, not low, middle and high "
            "income tiers. A result in the referral band between {lower} and {upper} means the "
            "model cannot separate the case. Records that land there split almost "
            "evenly, {referred_actual_rate} of them genuinely above the threshold, "
            "and the model is only {referred_accuracy} accurate on them, which is "
            "close to chance. That is why those cases are escalated to a person "
            "instead of decided automatically."
        ),
    },
    {
        "key": "accuracy",
        "heading": "The model is wrong regularly",
        "template": (
            "On {test_rows} held-out records the model is {accuracy} accurate overall and "
            "recovers {recall} of the people who actually fall in the upper bracket. Restricting "
            "automated decisions to cases outside the referral band raises accuracy to "
            "{automated_accuracy}, but that comes at the cost of sending {referral_rate} of cases "
            "to a human assessor, and it does not make the remaining decisions certain."
        ),
    },
    {
        "key": "group_differences",
        "heading": "Outcomes differ by sex",
        "template": (
            "On the held-out file the model predicts the upper bracket for {male_selection} of "
            "men and {female_selection} of women. Part of that reflects the underlying data, "
            "where the actual rates are {male_actual} and {female_actual}. The sharper figure is "
            "the error rate: among people who genuinely are above the threshold, the model "
            "identifies {male_tpr} of the men and {female_tpr} of the women. That is a disparity "
            "in who the model fails, not only in who it selects."
        ),
    },
    {
        "key": "counterfactual",
        "heading": "Changing only the sex field changes the answer",
        "template": (
            "Reversing the sex field on every held-out record and rescoring changes {flipped} "
            "predictions, {flipped_share} of the file. Of those, {men_losing} are men who lose "
            "their upper bracket prediction when relabelled female and {women_gaining} are women "
            "who gain one when relabelled male; {other_direction} move in other directions. The "
            "field is not decorative. It moves outcomes on its own."
        ),
    },
    {
        "key": "proxies",
        "heading": "Removing protected attributes does not remove the problem",
        "template": (
            "Retraining with sex, race and country of birth deleted removes {columns_removed} "
            "prepared columns and barely changes anything. Accuracy moves from {accuracy_with} to "
            "{accuracy_without}, and the gap in selection rates between men and women falls only "
            "from {gap_with} to {gap_without}, closing about {gap_share_closed} of it. The reason "
            "is that other columns carry the same information: a classifier trained to predict "
            "sex from the remaining features is {proxy_with_role} accurate against a "
            "{proxy_baseline} baseline, and still {proxy_without_role} accurate once the "
            "household role column is removed as well. This is the proxy variable problem "
            "described by Barocas and Selbst. Not collecting an attribute is not a defence "
            "against discriminating on it."
        ),
    },
    {
        "key": "defaults",
        "heading": "Unanswered fields still affect the result",
        "template": (
            "The form does not ask for all thirteen attributes. Anything left blank is filled "
            "with the training median or mode, and those filled values contribute to the "
            "prediction like any other. Capital gains is the clearest case: its default is zero, "
            "zero is informative to the model, and it pushes the prediction away from the upper "
            "bracket. The fields that were defaulted are listed with every result so that a "
            "partial profile is never mistaken for a complete one."
        ),
    },
    {
        "key": "legal",
        "heading": "This must not be used to decide about real people",
        "template": (
            "This is a demonstration of a decision support pattern built for an academic "
            "assessment. Using it to screen applicants for lending, employment, housing or any "
            "similar decision would raise indirect discrimination questions under sections 19 and "
            "29 of the Equality Act 2010, and the evidence above indicates it would not withstand "
            "that scrutiny. The referral band exists so that cases the model cannot separate reach "
            "a person rather than an automated verdict, but that safeguard does not make the "
            "underlying disparities acceptable."
        ),
    },
]


def panel_text():
    """Return the limitations as a list of heading and body pairs, in order."""
    values = figures()
    return [
        {"key": s["key"], "heading": s["heading"], "body": s["template"].format(**values)}
        for s in SECTIONS
    ]


def section(key):
    """Return one limitations section by key, for the conversational layer."""
    for item in panel_text():
        if item["key"] == key:
            return item
    return None


def summary_paragraph():
    """Return a single short paragraph for use where the full panel does not fit."""
    values = figures()
    return (
        "This model was trained on 1994 US census data against a threshold that has not been "
        "adjusted since. Its outcomes differ by sex, and deleting the protected attributes closes "
        "only about {gap_share_closed} of that gap because other columns carry the same "
        "information. It is a demonstration of a decision support pattern, not a tool for making "
        "decisions about real people."
    ).format(**values)


if __name__ == "__main__":
    for item in panel_text():
        print(f"\n{item['heading']}")
        print("-" * len(item["heading"]))
        print(item["body"])

    print("\n\nSummary")
    print("-------")
    print(summary_paragraph())