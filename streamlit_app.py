"""Income bracket decision support chatbot.

Run from the repository root with ``streamlit run streamlit_app.py``.

The interface loads artefacts written by ``src.train`` and never fits a model.
Artefact loading is cached so it happens once per session rather than on every
widget interaction, since Streamlit re-executes this script top to bottom each
time the user touches anything.

Sections marked below are filled in by later phases.
"""

import streamlit as st

from src import config
from src.chatbot import limitations
from src.chatbot.engine import question_buttons
from src.decision import decide, load_thresholds
from src.nlp import ensure_corpora
from src.predict import load_artefacts, score_person, version_report

st.set_page_config(
    page_title="Income bracket decision support",
    page_icon="◱",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource(show_spinner="Loading the trained models")
def startup():
    """Load every artefact once and fetch the language corpora.

    Cached as a resource rather than data because these objects are shared
    across sessions and are not serialisable in any useful sense. The corpora
    download runs here so that it happens on first use rather than on import,
    which is what stops a deployed application failing on its first message.
    """
    ensure_corpora()

    artefacts = load_artefacts()
    load_thresholds()

    return {
        "metadata": artefacts["metadata"],
        "warnings": version_report(),
    }


def initialise_state():
    """Create the session keys the interface depends on."""
    st.session_state.setdefault("assessment", None)
    st.session_state.setdefault("person_name", "")
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("untouched_fields", [])


def header(metadata):
    """Draw the title and the standing caveat."""
    st.title("Income bracket decision support")
    st.caption(
        "Predicts whether a person falls above or below the 1994 US census income "
        "threshold, explains which attributes drove that prediction, and routes "
        "uncertain cases to a human assessor. Built for COM727. Not a tool for "
        "deciding anything about a real person."
    )

    columns = st.columns(4)
    held_out = metadata["metrics"]["held_out_test"]
    columns[0].metric("Held-out accuracy", f"{held_out['accuracy']:.1%}")
    columns[1].metric("Recall", f"{held_out['recall']:.1%}")
    columns[2].metric("Records held back", f"{metadata['rows']['held_out_test']:,}")
    columns[3].metric("Attributes used", metadata["feature_count"])


def show_warnings(warnings):
    """Surface a library version mismatch rather than letting it act silently."""
    for warning in warnings:
        st.warning(
            f"{warning}. The saved preprocessing objects were written by a "
            "different version, so results may not match those reported.",
            icon="⚠",
        )


BAND_STYLE = {
    "below": ("Likely below the threshold", "#5A6B7C"),
    "referral": ("Refer for human assessment", "#8C4A1F"),
    "above": ("Likely above the threshold", "#2E7D5B"),
}


def category_options(field):
    """Return the values this field took in the training data, in a sensible order."""
    levels = load_artefacts()["schema"]["category_levels"][field]

    if field == "education":
        return sorted(levels, key=lambda level: config.EDUCATION_NUM.get(level, 0))
    return levels


def field_input(field, defaults, container):
    """Draw the right control for one attribute and return what was entered."""
    label = config.FIELD_LABELS[field]

    if field in config.NUMERIC_RANGES:
        low, high = config.NUMERIC_RANGES[field]
        return container.number_input(
            label, min_value=low, max_value=high,
            value=int(defaults[field]), step=1, key=f"field_{field}")

    options = category_options(field)
    return container.selectbox(
        label, options, index=options.index(defaults[field]),
        key=f"field_{field}")


def assessment_form():
    """Draw the form and return the answers, or None if nothing was submitted."""
    defaults = load_artefacts()["defaults"]

    st.session_state.person_name = st.text_input(
        "Name", value=st.session_state.person_name,
        placeholder="Used for display only, never sent to the model")

    answers = {}
    columns = st.columns(2)
    for position, field in enumerate(config.PRIMARY_FIELDS):
        answers[field] = field_input(field, defaults, columns[position % 2])

    with st.expander("Further details, filled from the training data if left alone"):
        st.caption(
            "Capital gains is worth setting deliberately. Its default is zero, and "
            "zero is a meaningful signal to the model rather than an absence of one."
        )
        optional = st.columns(2)
        for position, field in enumerate(config.OPTIONAL_FIELDS):
            answers[field] = field_input(field, defaults, optional[position % 2])

    return answers if st.button("Assess", type="primary") else None


def show_outcome(assessment):
    """Report the band, the recommended action and what was assumed."""
    outcome = decide(assessment["probability"])
    heading, colour = BAND_STYLE[outcome["band"]]
    name = st.session_state.person_name.strip() or "This person"

    st.markdown(
        f"<div style='border-left:6px solid {colour};padding:0.6rem 1rem;"
        f"background:#F1F4F7;border-radius:4px'>"
        f"<div style='font-size:1.35rem;font-weight:600;color:{colour}'>{heading}</div>"
        f"<div style='margin-top:0.3rem'>{outcome['action']}</div></div>",
        unsafe_allow_html=True,
    )

    st.write("")
    columns = st.columns([2, 3])
    columns[0].metric("Probability of the upper bracket",
                      f"{assessment['probability']:.1%}")
    columns[0].caption(f"Band covers {outcome['range']}")
    columns[1].write(f"**{name}.** {outcome['interpretation']}")

    st.progress(min(max(assessment["probability"], 0.0), 1.0))

    untouched = st.session_state.untouched_fields
    if untouched:
        listed = ", ".join(config.FIELD_LABELS.get(f, f).lower() for f in untouched)
        st.caption(
            f"Left at the training default: {listed}. These contribute to the result "
            "like any other answer, so a default is an assumption rather than a blank."
        )
    else:
        st.caption("Every field was set deliberately. Nothing was assumed.")


def assessment_tab():
    """Enter a person and see the decision band."""
    st.subheader("Describe a person")
    st.caption(
        "Every field maps to a value present in the 1994 census data. Years of "
        "education are taken from the qualification rather than asked for."
    )

    answers = assessment_form()

    if answers is not None:
        defaults = load_artefacts()["defaults"]
        st.session_state.untouched_fields = [
            field for field in config.OPTIONAL_FIELDS
            if answers[field] == defaults[field]
        ]
        st.session_state.assessment = score_person(answers)
        st.session_state.messages = []

    if st.session_state.assessment is None:
        st.info("Fill in the details and select Assess.")
        return

    st.divider()
    show_outcome(st.session_state.assessment)


def explanation_tab():
    """Show the per attribute contribution breakdown."""
    if st.session_state.assessment is None:
        st.info("Assess someone first and the breakdown will appear here.")
        return
    st.info("The contribution chart is added in Phase 7.3.")


def conversation_tab():
    """Ask questions about the assessment, the model and the data."""
    ready = st.session_state.assessment is not None
    grouped = question_buttons(ready)
    available = sum(len(v) for v in grouped.values())

    st.caption(
        f"{available} questions available"
        + ("" if ready else ", and more once someone has been assessed")
    )
    st.info("The conversation is wired up in Phase 7.4.")


def model_tab():
    """Report how the model performs and where it should not be trusted."""
    st.info("The comparison table is added in Phase 7.5.")

    st.subheader("Limitations")
    for section in limitations.panel_text():
        with st.expander(section["heading"]):
            st.write(section["body"])


def main():
    """Draw the application."""
    context = startup()
    initialise_state()

    header(context["metadata"])
    show_warnings(context["warnings"])

    tabs = st.tabs(["Assessment", "Why", "Ask", "Model and limitations"])

    with tabs[0]:
        assessment_tab()
    with tabs[1]:
        explanation_tab()
    with tabs[2]:
        conversation_tab()
    with tabs[3]:
        model_tab()

    st.divider()
    trained = context["metadata"]["trained_at"][:10]
    st.caption(
        f"Model trained {trained} on the UCI Adult dataset, 1994 US Census. "
        "Predictions describe patterns in that data and are not advice about "
        "anyone's earnings."
    )


if __name__ == "__main__":
    main()