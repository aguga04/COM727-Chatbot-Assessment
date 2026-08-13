"""Income bracket decision support chatbot.

Run from the repository root with ``streamlit run streamlit_app.py``.

The interface loads artefacts written by ``src.train`` and never fits a model.
Artefact loading is cached so it happens once per session rather than on every
widget interaction, since Streamlit re-executes this script top to bottom each
time the user touches anything.

Sections marked below are filled in by later phases.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.chatbot import limitations
from src.chatbot.engine import READABLE, question_buttons, respond
from src.decision import decide, load_thresholds
from src.explain import explain
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


TOWARDS_COLOUR = "#2E7D5B"
AWAY_COLOUR = "#B5533C"


def format_attribute_value(value):
    """Return an attribute value as text.

    The column mixes whole numbers with category names, which cannot be handed
    to the table renderer as it stands, and a whole number should not gain a
    decimal point on the way to the screen.
    """
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        return str(int(number)) if number.is_integer() else f"{number:.2f}"
    return str(value)


def contribution_chart(table):
    """Draw the per attribute contributions as a diverging horizontal bar chart."""
    ordered = table.iloc[::-1]

    labels = [READABLE.get(a, a) for a in ordered["attribute"]]
    values = ordered["contribution"].tolist()
    colours = [TOWARDS_COLOUR if v >= 0 else AWAY_COLOUR for v in values]
    hover = [
        f"{label}: {format_attribute_value(value)}"
        f"<br>contribution {contribution:+.3f} log odds"
        for label, value, contribution in zip(labels, ordered["value"], values)
    ]

    figure = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colours, hovertext=hover, hoverinfo="text",
        text=[f"{v:+.3f}" for v in values], textposition="outside", cliponaxis=False,
    ))

    span = max(abs(min(values)), abs(max(values))) * 1.35
    figure.update_layout(
        height=32 * len(labels) + 90,
        margin=dict(l=10, r=10, t=10, b=40),
        xaxis=dict(title="Contribution in log odds", range=[-span, span],
                   zeroline=True, zerolinewidth=1, zerolinecolor="#666666"),
        yaxis=dict(title=None),
        plot_bgcolor="#FFFFFF",
        showlegend=False,
    )
    return figure


def arithmetic_block(explanation, assessment):
    """Show that the contributions reconstruct the model output exactly."""
    total = float(explanation["table"]["contribution"].sum())

    st.markdown(
        f"""
| Step | Value |
| --- | ---: |
| Model baseline before any attribute is considered | {explanation['bias']:+.4f} |
| Sum of the thirteen contributions above | {total:+.4f} |
| Log odds for this person | {explanation['log_odds']:+.4f} |
| Probability after converting the log odds | {explanation['probability']:.4f} |
| Probability reported by the model itself | {assessment['probability']:.4f} |
"""
    )
    st.caption(
        f"The two probabilities agree to within {explanation['reconstruction_error']:.1e}. "
        "The attribution is exact rather than an estimate, which is why it can be "
        "checked rather than trusted."
    )


def explanation_tab():
    """Show the per attribute contribution breakdown."""
    if st.session_state.assessment is None:
        st.info("Assess someone on the first tab and the breakdown will appear here.")
        return

    assessment = st.session_state.assessment
    explanation = explain(assessment)
    table = explanation["table"]

    st.subheader("What drove this prediction")
    st.caption(
        "Each bar is one attribute's exact share of the model's output, measured in "
        "log odds. Green pushes towards the upper bracket, red pushes away. A "
        "contribution of +0.8 does not mean eighty percent and does not mean any "
        "sum of money."
    )

    towards = table[table["contribution"] > 0]
    away = table[table["contribution"] < 0]
    columns = st.columns(2)
    if not towards.empty:
        row = towards.iloc[0]
        columns[0].metric(
            "Strongest push towards",
            READABLE.get(row["attribute"], row["attribute"]),
            f"{row['contribution']:+.3f}")
    if not away.empty:
        row = away.iloc[0]
        columns[1].metric(
            "Strongest push away",
            READABLE.get(row["attribute"], row["attribute"]),
            f"{row['contribution']:+.3f}")

    st.plotly_chart(contribution_chart(table), width="stretch")

    with st.expander("How this adds up"):
        arithmetic_block(explanation, assessment)

    with st.expander("Every attribute as a table"):
        display = table.copy()
        display["attribute"] = [READABLE.get(a, a) for a in display["attribute"]]
        display["value"] = display["value"].map(format_attribute_value)
        display["contribution"] = display["contribution"].map(lambda v: f"{v:+.3f}")
        display = display.rename(columns={
            "attribute": "Attribute", "value": "Value",
            "contribution": "Contribution (log odds)", "direction": "Direction"})
        st.dataframe(display, width="stretch", hide_index=True)


CATEGORY_ORDER = ["This prediction", "Why", "The model", "The data",
                  "Limitations", "Conversation"]


def ask(question):
    """Send a question through the classifier and record both sides of the exchange."""
    outcome = respond(question, assessment=st.session_state.assessment)

    st.session_state.messages.append({"role": "user", "text": question})
    st.session_state.messages.append({
        "role": "assistant",
        "text": outcome["response"],
        "tag": outcome["tag"],
        "confidence": outcome["confidence"],
        "threshold": outcome["threshold"],
        "used_fallback": outcome["used_fallback"],
    })


def replay_history():
    """Draw the conversation so far, with the classifier's working shown."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["text"])

            if message["role"] == "assistant":
                label = (
                    f"Fallback, confidence {message['confidence']:.3f} "
                    f"below the {message['threshold']:.2f} threshold"
                    if message["used_fallback"]
                    else f"Recognised as {message['tag']}, "
                         f"confidence {message['confidence']:.3f}"
                )
                with st.expander(label):
                    st.caption(
                        "The question was classified by a Multinomial Naive Bayes "
                        "model trained on hand written examples. Anything below "
                        f"{message['threshold']:.2f} confidence returns a fallback "
                        "instead of an answer."
                    )


def question_panel(grouped, ready):
    """Offer the available questions, grouped by topic."""
    for category in CATEGORY_ORDER:
        questions = grouped.get(category)
        if not questions:
            continue

        expanded = category == ("This prediction" if ready else "The model")
        with st.expander(f"{category}  ({len(questions)})", expanded=expanded):
            columns = st.columns(2)
            for position, question in enumerate(questions):
                if columns[position % 2].button(
                        question["display"], key=f"ask_{question['tag']}",
                        width="stretch"):
                    ask(question["display"])
                    st.rerun()


def conversation_tab():
    """Ask questions about the assessment, the model and the data."""
    ready = st.session_state.assessment is not None
    grouped = question_buttons(ready)
    available = sum(len(questions) for questions in grouped.values())

    st.subheader("Ask about this model")
    st.caption(
        f"{available} questions available"
        + ("." if ready else ", and more once someone has been assessed.")
        + " Selecting one sends that sentence to the intent classifier exactly as "
        "typed input would arrive. The wording on each button was held out of "
        "training, so recognising it is a genuine classification rather than a lookup."
    )

    if st.session_state.messages:
        replay_history()
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.rerun()
        st.divider()

    question_panel(grouped, ready)


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