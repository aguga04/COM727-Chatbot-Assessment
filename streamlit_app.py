"""Income bracket decision support chatbot.

Run from the repository root with ``streamlit run streamlit_app.py``.

The interface loads artefacts written by ``src.train`` and never fits a model.
Artefact loading is cached so it happens once per session rather than on every
widget interaction, since Streamlit re-executes this script top to bottom each
time the user touches anything.

Sections marked below are filled in by later phases.
"""

import streamlit as st

from src.chatbot import limitations
from src.chatbot.engine import question_buttons
from src.decision import load_thresholds
from src.nlp import ensure_corpora
from src.predict import load_artefacts, version_report

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


def assessment_tab():
    """Enter a person and see the decision band."""
    st.info("The assessment form is added in Phase 7.2.")


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