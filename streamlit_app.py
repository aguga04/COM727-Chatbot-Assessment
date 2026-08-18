"""Income bracket decision support chatbot.

Run from the repository root with ``streamlit run streamlit_app.py``.

The interface loads artefacts written by ``src.train`` and never fits a model.
Artefact loading is cached so it happens once per session rather than on every
widget interaction, since Streamlit re-executes this script top to bottom each
time the user touches anything.

Sections marked below are filled in by later phases.
"""

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.chatbot import limitations
from src.chatbot.engine import (READABLE, format_probability,
                                question_buttons, respond)
from src.decision import band_evidence, decide, evidence_sentence, load_thresholds
from src.explain import explain
from src.nlp import ensure_corpora
from src.predict import load_artefacts, score_person, version_report

st.set_page_config(
    page_title="Nexus Income Bracket Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# The interface uses a single light palette. Text colour is never set inline, so
# every label follows the theme Streamlit is rendering and cannot end up dark on
# dark or pale on white.
COLOURS = {
    "surface": "#F1F4F7",
    "accent": "#1B3A5C",
    "muted": "#5A6B7C",
    "towards": "#2E7D5B",
    "away": "#B5533C",
    "suit": "#1B3A5C",
    "body": "#C8D2DB",
    "visor": "#0E1F30",
    "coin": "#D9A441",
    "background": "#FFFFFF",
}


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


def robot():
    """Return the mascot as an SVG string.

    Drawn rather than loaded so it needs no asset file and scales cleanly. It is
    handed to ``st.image`` rather than to ``st.markdown``: markdown strips svg
    elements during sanitising, which leaves only the loose text inside them.
    Motion is dropped for anyone who has asked their system to reduce it.
    """
    colours = COLOURS
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" width="120" height="120">
  <style>
    @keyframes nexus-bob {{ 0%,100% {{ transform: translateY(0); }}
                            50% {{ transform: translateY(-3px); }} }}
    @keyframes nexus-blink {{ 0%,92%,100% {{ transform: scaleY(1); }}
                              96% {{ transform: scaleY(0.08); }} }}
    @keyframes nexus-spin {{ 0%,100% {{ transform: scaleX(1); }}
                             50% {{ transform: scaleX(0.25); }} }}
    @keyframes nexus-pulse {{ 0%,100% {{ opacity: 0.45; }} 50% {{ opacity: 1; }} }}
    .nx-body {{ animation: nexus-bob 4s ease-in-out infinite; transform-origin: 60px 60px; }}
    .nx-eye  {{ animation: nexus-blink 5.5s ease-in-out infinite; transform-origin: center; }}
    .nx-coin {{ animation: nexus-spin 3.2s ease-in-out infinite; transform-origin: 96px 74px; }}
    .nx-tip  {{ animation: nexus-pulse 2.4s ease-in-out infinite; }}
    @media (prefers-reduced-motion: reduce) {{
      .nx-body, .nx-eye, .nx-coin, .nx-tip {{ animation: none; }}
    }}
  </style>
  <g class="nx-body">
    <line x1="60" y1="22" x2="60" y2="12" stroke="{colours['body']}" stroke-width="3"/>
    <circle class="nx-tip" cx="60" cy="9" r="4" fill="{colours['coin']}"/>
    <rect x="34" y="22" width="52" height="38" rx="12" fill="{colours['body']}"/>
    <rect x="41" y="32" width="38" height="19" rx="8" fill="{colours['visor']}"/>
    <circle class="nx-eye" cx="52" cy="41" r="3.6" fill="{colours['coin']}"/>
    <circle class="nx-eye" cx="68" cy="41" r="3.6" fill="{colours['coin']}"/>
    <path d="M60 62 L44 70 L44 106 L76 106 L76 70 Z" fill="{colours['suit']}"/>
    <path d="M60 62 L52 68 L60 84 L68 68 Z" fill="{colours['background']}"/>
    <path d="M60 66 L56 71 L60 79 L64 71 Z" fill="{colours['coin']}"/>
    <rect x="30" y="72" width="12" height="30" rx="6" fill="{colours['suit']}"/>
    <rect x="78" y="72" width="12" height="30" rx="6" fill="{colours['suit']}"/>
    <g class="nx-coin">
      <circle cx="96" cy="74" r="11" fill="{colours['coin']}"/>
      <text x="96" y="79" font-size="13" font-weight="700" text-anchor="middle"
            fill="{colours['visor']}">$</text>
    </g>
  </g>
</svg>
"""


def header(metadata):
    """Draw the mascot, the title and the headline figures."""
    left, right = st.columns([1, 9], vertical_alignment="center")

    with left:
        st.image(robot(), width=110)

    with right:
        st.markdown(
            "<div style='font-size:2.1rem;font-weight:700;line-height:1.1;"
            "color:inherit'>Nexus</div>"
            "<div style='font-size:1.2rem;font-weight:500;color:inherit;"
            "opacity:0.85'>Income Bracket Chatbot</div>",
            unsafe_allow_html=True)

    st.caption(
        "Predicts whether a person falls above or below the 1994 US census income "
        "threshold, explains which attributes drove that prediction, and routes "
        "uncertain cases to a human assessor. Built for COM727 by team Nexus. Not a "
        "tool for deciding anything about a real person."
    )

    held_out = metadata["metrics"]["held_out_test"]
    columns = st.columns(4)
    columns[0].metric(
        "Held-out accuracy", f"{held_out['accuracy']:.1%}",
        help="Of every 100 people the model has never seen, it puts this many on the "
             "correct side of the income threshold.")
    columns[1].metric(
        "Recall", f"{held_out['recall']:.1%}",
        help="Of the people who genuinely are above the threshold, this share is "
             "correctly identified. The rest are missed.")
    rows = metadata["rows"]
    columns[2].metric(
        "Data split",
        f"{rows['train']:,} / {rows['validation']:,} / {rows['held_out_test']:,}",
        help="Training, validation, and held back for final testing. The held-back "
             "records were never seen during training, so the accuracy shown here is "
             "not flattered by memorisation.")
    columns[3].metric(
        "Attributes used", metadata["feature_count"],
        help="Thirteen real world attributes become this many numeric columns once "
             "categories such as occupation are split out.")


def show_warnings(warnings):
    """Surface a library version mismatch rather than letting it act silently."""
    for warning in warnings:
        st.warning(
            f"{warning}. The saved preprocessing objects were written by a "
            "different version, so results may not match those reported.",
            icon="⚠",
        )


# A bare dollar sign starts LaTeX maths in Streamlit markdown, which swallows the
# text between two of them. Escaping keeps the symbol literal.
UPPER_LABEL = r"Upper bracket, over \$50,000"
LOWER_LABEL = r"Lower bracket, \$50,000 or under"

# Bands are named by what the model concluded, not by an action taken against a
# person. Calling them accept and reject would claim the system makes lending
# decisions, which is exactly what the limitations panel says it must not do.
# The credit screening analogy is named once in prose beneath instead.
BAND_STYLE = {
    "below": {
        "heading": "Lower bracket",
        "colour": "#1B3A5C",
        "strip": "Confident enough to answer automatically.",
    },
    "referral": {
        "heading": "Refer for human assessment",
        "colour": "#8C4A1F",
        "strip": "Too close to call. This case needs a person.",
    },
    "above": {
        "heading": "Upper bracket",
        "colour": "#2E7D5B",
        "strip": "Confident enough to answer automatically.",
    },
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


def tint(hex_colour, alpha):
    """Return a hex colour as a translucent rgba string."""
    red, green, blue = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({red},{green},{blue},{alpha})"


def band_strip(probability, band):
    """Draw the three bands as one bar, marking where this person falls.

    Makes the routing visible rather than implied. The reader can see that the
    system has three outcomes, where the boundaries sit, and which side of them
    this case landed on.

    Inactive segments keep their colour as text on a pale tint rather than being
    faded out, because a faded segment on a white page loses its edges and the
    reader can no longer see that three bands exist.
    """
    thresholds = load_thresholds()
    low, high = thresholds["lower"], thresholds["upper"]
    marker = min(max(probability, 0.0), 1.0) * 100

    segments = [
        ("below", "Lower bracket", round(low * 100, 2)),
        ("referral", "Refer to a person", round((high - low) * 100, 2)),
        ("above", "Upper bracket", round((1 - high) * 100, 2)),
    ]

    cells = ""
    for key, name, width in segments:
        colour = BAND_STYLE[key]["colour"]
        active = key == band
        cells += (
            f"<div style='flex:0 0 {width}%;padding:7px 4px;text-align:center;"
            f"font-size:0.78rem;font-weight:{'700' if active else '600'};"
            f"color:{'#FFFFFF' if active else colour};"
            f"background:{colour if active else tint(colour, 0.16)};"
            f"border-right:1px solid rgba(255,255,255,0.85)'>{name}</div>"
        )

    ticks = "".join(
        f"<span style='position:absolute;left:{position}%;{shift};"
        f"font-size:0.7rem;opacity:0.65'>{label}</span>"
        for position, label, shift in [
            (0, "0%", "transform:translateX(0)"),
            (low * 100, f"{low:.0%}", "transform:translateX(-50%)"),
            (high * 100, f"{high:.0%}", "transform:translateX(-50%)"),
            (100, "100%", "transform:translateX(-100%)"),
        ]
    )

    st.markdown(
        f"<div style='margin:0.4rem 0 0.2rem 0'>"
        f"<div style='display:flex;border-radius:5px;overflow:hidden;"
        f"border:1px solid rgba(0,0,0,0.12)'>{cells}</div>"
        f"<div style='position:relative;height:19px'>"
        f"<span style='position:absolute;left:{marker}%;transform:translateX(-50%);"
        f"font-size:0.74rem;font-weight:600'>&#9650; {probability:.1%}</span></div>"
        f"<div style='position:relative;height:14px'>{ticks}</div></div>",
        unsafe_allow_html=True,
    )


def leaning_sentence(probability, band):
    """Say in one line what the binary prediction is and how firmly it is held.

    The percentage is formatted so that a very small probability does not round
    to zero, which would read as an impossibility rather than as a small number.
    """
    if band == "referral":
        leaning = "upper" if probability >= 0.5 else "lower"
        return (
            f"That leans towards the {leaning} bracket, but not firmly enough to act "
            f"on, so this case is referred rather than answered."
        )

    if probability >= 0.5:
        return f"That is a clear enough lean, so its answer is {UPPER_LABEL.lower()}."

    return f"That is a clear enough lean, so its answer is {LOWER_LABEL.lower()}."


def result_banner(name, colour):
    """Draw the centred heading that opens the result."""
    st.markdown(
        f"<div style='background:{colour};border-radius:6px;padding:0.55rem 1rem;"
        f"text-align:center;margin:0.2rem 0 0.9rem 0'>"
        f"<span style='color:#FFFFFF;font-size:1.25rem;font-weight:700;"
        f"letter-spacing:0.01em'>Results for {name}</span></div>",
        unsafe_allow_html=True,
    )


def probability_card(probability, colour):
    """Show the figure the whole result turns on, at a size that reads across a room."""
    st.markdown(
        f"<div style='border:1px solid {tint(colour, 0.35)};border-radius:6px;"
        f"background:{tint(colour, 0.08)};padding:1.1rem 1rem;text-align:center;"
        f"height:100%'>"
        f"<div style='font-size:0.86rem;opacity:0.75;line-height:1.3'>"
        f"The model puts the chance of the</div>"
        f"<div style='font-size:0.95rem;font-weight:600;opacity:0.85;"
        f"margin-bottom:0.35rem'>upper bracket at</div>"
        f"<div style='font-size:3.4rem;font-weight:700;line-height:1;"
        f"color:{colour}'>{format_probability(probability)}</div></div>",
        unsafe_allow_html=True,
    )


def verdict_card(style):
    """Show the routing decision and the action it calls for."""
    st.markdown(
        f"<div style='border-left:6px solid {style['colour']};"
        f"border-radius:6px;background:{COLOURS['surface']};"
        f"padding:1.1rem 1.2rem;height:100%'>"
        f"<div style='font-size:1.3rem;font-weight:700;color:{style['colour']};"
        f"margin-bottom:0.4rem'>{style['heading']}</div>"
        f"<div style='line-height:1.45'>{style['action']}</div>"
        f"<div style='margin-top:0.5rem;font-size:0.85rem;opacity:0.75'>"
        f"{style['strip']}</div></div>",
        unsafe_allow_html=True,
    )


def show_outcome(assessment):
    """Report the band, the recommended action and what was assumed."""
    probability = assessment["probability"]
    outcome = decide(probability)
    style = dict(BAND_STYLE[outcome["band"]])
    style["action"] = outcome["action"]
    name = st.session_state.person_name.strip() or "this person"
    thresholds = load_thresholds()

    result_banner(name, style["colour"])

    left, right = st.columns(2, gap="medium")
    with left:
        probability_card(probability, style["colour"])
    with right:
        verdict_card(style)

    st.write("")
    st.markdown(f"**{leaning_sentence(probability, outcome['band'])}**")

    st.markdown(
        "This is a two way choice. Every person is automatically placed in one of "
        f"two brackets recorded by the 1994 census: {UPPER_LABEL.lower()}, or "
        f"{LOWER_LABEL.lower()}. The percentage above is how sure the model is of "
        "the first of those, so anything under 50 percent leans towards the second. "
        "Bands exist for the borderline cases in the middle, where the percentage "
        "is not decisive enough for the system to place someone on its own."
    )

    st.markdown(
        f"**What the bands are.** A band is a range of confidence, not a range of "
        f"income. The percentage falls somewhere between 0 and 100, and we divide "
        f"that line into three parts. Below {thresholds['lower']:.0%} and above "
        f"{thresholds['upper']:.0%} the model is sure enough to answer on its own. "
        f"In between it is not, so the case goes to a person instead. The bar below "
        f"shows those three parts and where this case landed."
    )

    band_strip(probability, outcome["band"])

    st.caption(
        f"This mirrors the accept, refer and decline routing used in credit "
        f"screening, where confident cases are handled automatically and borderline "
        f"ones are passed to a human. The {thresholds['lower']:.0%} and "
        f"{thresholds['upper']:.0%} boundaries were chosen by measuring accuracy "
        "against how many cases each setting would send to a person, not by "
        "preference."
    )

    st.info(outcome["interpretation"], icon="ℹ")

    untouched = st.session_state.untouched_fields
    if untouched:
        listed = ", ".join(config.FIELD_LABELS.get(f, f).lower() for f in untouched)
        st.caption(
            f"Left at the training default: {listed}. These contribute to the result "
            "like any other answer, so a default is an assumption rather than a blank."
        )
    else:
        st.caption("Every field was set deliberately. Nothing was assumed.")

    st.caption(
        "This describes patterns in 1994 census data, not this person. It is not a "
        "statement about what anyone earns."
    )


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


def odds_phrase(contribution):
    """Translate a log odds contribution into everyday language.

    Exponentiating the contribution gives the factor it multiplies the odds by,
    which is the same figure without requiring anyone to know what a log odd is.
    The wording leads with direction and strength so a reader who wants only the
    gist can stop after three words, and the multiplier follows for anyone who
    wants the size. The word odds is avoided, since it is the term a non
    technical reader stumbles on.
    """
    size = abs(float(contribution))

    if size < 0.05:
        return "No real effect"

    strength = "Strong" if size >= 0.7 else "Moderate" if size >= 0.3 else "Slight"
    factor = float(np.exp(contribution))

    if contribution > 0:
        return f"{strength} push up, about {factor:.1f} times more likely"
    return f"{strength} push down, about {factor:.1f} times as likely"


def number_glossary():
    """Explain the figures on this page for a reader who does not work with them."""
    st.markdown(
        """
| Term | What it means in plain English |
| --- | --- |
| **Band** | A range of confidence, not a range of income. The scale from 0 to 100 percent is split into three parts: confident lower, uncertain, confident upper. Which part a case falls into decides whether the system answers it or refers it to a person. |
| **Probability** | How sure the model is, from 0 to 100 percent. Seventy percent means that out of a hundred similar people, about seventy were above the threshold. |
| **Log odds** | The unit the model works in internally. Zero means no effect. Positive pushes towards the upper bracket, negative pushes away. It is not a percentage and not an amount of money. |
| **Times more likely** | The same figure made readable. A contribution of +0.80 means the attribute made the upper bracket about 2.2 times more likely. Below 1.0 means less likely. |
| **Accuracy** | How often the model puts a person on the correct side of the threshold. |
| **Recall** | Of the people who really are above the threshold, how many the model finds. Low recall means it misses people. |
| **Precision** | When the model says upper bracket, how often it is right. |
| **F1** | Precision and recall combined into one number, so a model cannot look good by being cautious. |
| **ROC-AUC** | How well the model ranks people. One is perfect, 0.5 is a coin flip. |
| **Brier score** | How honest the percentages are. Zero is perfect, lower is better. |
| **Calibration error** | The average gap between what the model predicted and what actually happened. Small means a stated seventy percent really behaves like seventy percent. |
| **Selection rate** | How often the model predicts the upper bracket for a group. A gap between groups is not automatically unfair, but it always needs explaining. |
"""
    )


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
    colours = [COLOURS["towards"] if v >= 0 else COLOURS["away"] for v in values]
    hover = [
        f"{label}: {format_attribute_value(value)}"
        f"<br>contribution {contribution:+.3f} log odds"
        f"<br>{odds_phrase(contribution)}"
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
                   zeroline=True, zerolinewidth=1, zerolinecolor=COLOURS["muted"],
                   gridcolor=COLOURS["surface"]),
        yaxis=dict(title=None),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
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
            f"{row['contribution']:+.3f}",
            help=odds_phrase(row["contribution"]))
    if not away.empty:
        row = away.iloc[0]
        columns[1].metric(
            "Strongest push away",
            READABLE.get(row["attribute"], row["attribute"]),
            f"{row['contribution']:+.3f}",
            help=odds_phrase(row["contribution"]))

    st.plotly_chart(contribution_chart(table), width="stretch")

    with st.expander("Check how these figures add up (technical)"):
        st.caption(
            "The chart above shows what each attribute did. This shows that those "
            "figures rebuild the model's own output exactly rather than "
            "approximating it, which is what makes the explanation checkable."
        )
        arithmetic_block(explanation, assessment)

    with st.expander("Every attribute as a table"):
        display = table.copy()
        display["attribute"] = [READABLE.get(a, a) for a in display["attribute"]]
        display["value"] = display["value"].map(format_attribute_value)
        display["plain"] = display["contribution"].map(odds_phrase)
        display["contribution"] = display["contribution"].map(lambda v: f"{v:+.3f}")
        display = display.rename(columns={
            "attribute": "Attribute", "value": "Value",
            "contribution": "Contribution (log odds)",
            "plain": "In plain English", "direction": "Direction"})
        st.dataframe(
            display[["Attribute", "Value", "Contribution (log odds)",
                     "In plain English"]],
            width="stretch", hide_index=True)

    with st.expander("What do these numbers mean?"):
        number_glossary()


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


@st.cache_data(show_spinner=False)
def load_results():
    """Read the generated result files the interface reports from."""
    comparison = pd.read_csv(config.MODELS / "val_vs_test_all_models.csv")

    with open(config.MODELS / "intent_metrics.json") as handle:
        intent_metrics = json.load(handle)

    with open(config.MODELS / "fairness_report.json") as handle:
        fairness = json.load(handle)

    return comparison, intent_metrics, fairness


def algorithm_comparison(comparison, metadata):
    """Show the eight way comparison and the reason one was deployed."""
    st.subheader("Algorithm comparison")
    st.caption(
        "Eight algorithms were fitted on the same training split and evaluated on "
        "the same validation split, then on the held-out test file the models never "
        "saw. The deployed model is the strongest performer, not the most readable one."
    )

    ranked = comparison.sort_values("F1_val", ascending=False).copy()
    deployed = metadata["model"].replace("Classifier", "")

    display = ranked.rename(columns={
        "Model": "Algorithm",
        "Accuracy_val": "Accuracy (validation)", "F1_val": "F1 (validation)",
        "ROC_AUC_val": "ROC-AUC (validation)",
        "Accuracy_test": "Accuracy (held out)", "F1_test": "F1 (held out)",
        "ROC_AUC_test": "ROC-AUC (held out)"})

    st.dataframe(
        display.style.format(precision=4).apply(
            lambda row: ["background-color: #E8F2EA" if deployed in str(row["Algorithm"])
                         else "" for _ in row], axis=1),
        width="stretch", hide_index=True)

    best = ranked.iloc[0]
    runner_up = ranked.iloc[1]
    st.caption(
        f"{best['Model']} leads on validation F1 by "
        f"{best['F1_val'] - runner_up['F1_val']:.4f} over {runner_up['Model']}, and "
        f"holds that lead on the held-out file. A tree ensemble has no coefficients "
        "to read, so the interpretability a linear model would have given for free is "
        "recovered through exact attribution instead."
    )


def decision_band_summary():
    """Explain where the band boundaries came from and how well calibrated they are."""
    evidence = band_evidence()

    st.subheader("Decision bands")
    st.write(evidence_sentence())

    columns = st.columns(4)
    columns[0].metric(
        "Referred to a person", f"{evidence['referral_rate']:.1%}",
        help="Share of cases the model declines to decide, sending them to a human "
             "assessor instead.")
    columns[1].metric(
        "Accuracy when automated", f"{evidence['automated_accuracy']:.1%}",
        f"{evidence['accuracy_gain']:+.1%}",
        help="Accuracy on the cases that are decided automatically. The change shown "
             "is the improvement over deciding everything with a single cut off.")
    columns[2].metric(
        "Brier score", f"{evidence['brier_score']:.4f}",
        help="How honest the percentages are. Zero is perfect. Always predicting "
             "fifty percent would score 0.25.")
    columns[3].metric(
        "Calibration error", f"{evidence['expected_calibration_error']:.4f}",
        help="Average gap between what the model predicted and what actually "
             "happened. Small means a stated seventy percent behaves like seventy "
             "percent.")
    st.caption(
        "A low calibration error means a predicted probability can be read as a "
        "probability rather than only as a ranking, which is what allows the bands "
        "to be defined on it at all."
    )


def conversational_summary(intent_metrics):
    """Report the second model, which is evaluated separately from the first."""
    st.subheader("The conversational model")
    st.caption(
        "A separate Multinomial Naive Bayes classifier decides what a question is "
        "asking before the income model is consulted. It is trained on hand written "
        "example phrasings and evaluated on its own terms."
    )

    columns = st.columns(4)
    columns[0].metric(
        "Intents", intent_metrics["intents"],
        help="Distinct questions the chatbot recognises.")
    columns[1].metric(
        "Training phrasings", intent_metrics["patterns"],
        help="Hand written example wordings the classifier learned from.")
    columns[2].metric(
        "Button accuracy", f"{intent_metrics['display_accuracy']:.0%}",
        help="Share of the question buttons that reach the right answer. None of "
             "those wordings appeared in training.")
    columns[3].metric(
        "Unseen phrasing accuracy", f"{intent_metrics['out_of_fold_accuracy']:.0%}",
        help="How often an arbitrary rewording reaches the right intent. Much lower, "
             "because the questions are numerous and similar to each other.")

    st.caption(
        f"The two accuracy figures measure different things and both are reported. "
        f"Button accuracy covers the {intent_metrics['intents']} question wordings the "
        "interface actually produces, none of which appeared in training. Unseen "
        "phrasing accuracy is cross validated across all "
        f"{intent_metrics['patterns']} training examples and is far lower, because the "
        "intents are numerous and closely related. Free text input is not offered, "
        "which is one reason why."
    )

    with st.expander("Per intent performance"):
        per_intent = pd.DataFrame(intent_metrics["per_intent"])
        st.dataframe(per_intent.style.format(precision=3),
                     width="stretch", hide_index=True)


def fairness_summary(fairness):
    """Show the audit findings that the limitations rest on."""
    rates = {r["group"]: r for r in fairness["group_rates"]["by_sex"]}
    flip = fairness["counterfactual_sex_only"]
    removal = fairness["attribute_removal"]
    proxy = fairness["proxy_recovery"]

    st.subheader("Fairness audit")

    columns = st.columns(3)
    columns[0].metric(
        "Predicted upper bracket, men against women",
        f"{rates['Male']['selection_rate']:.1%} / {rates['Female']['selection_rate']:.1%}")
    columns[1].metric(
        "Predictions changed by reversing sex alone",
        f"{flip['predictions_changed']:,}",
        f"{flip['share_changed']:.2%} of the held-out file")
    columns[2].metric(
        "Sex recovered from the other columns",
        f"{proxy['household_role_present']['accuracy']:.1%}",
        f"baseline {proxy['majority_class_baseline']:.1%}")

    gap_with = removal["with_protected_attributes"]["selection_rate_gap"]
    gap_without = removal["without_protected_attributes"]["selection_rate_gap"]
    st.caption(
        f"Retraining without sex, race and country of birth moves the gap in selection "
        f"rates from {gap_with:.3f} to {gap_without:.3f}, closing about "
        f"{removal['gap_closed'] / gap_with:.0%} of it, while accuracy barely moves. "
        "Other columns carry the same information, so not collecting an attribute is "
        "not a defence against discriminating on it."
    )


def model_tab():
    """Report how the model performs and where it should not be trusted."""
    comparison, intent_metrics, fairness = load_results()
    metadata = load_artefacts()["metadata"]

    algorithm_comparison(comparison, metadata)
    st.divider()
    decision_band_summary()
    st.divider()
    conversational_summary(intent_metrics)
    st.divider()
    fairness_summary(fairness)
    st.divider()

    st.subheader("Limitations")
    st.caption(
        "Every figure on this page is read from files the training and audit scripts "
        "generate. Retraining the model changes what this page says."
    )
    for section in limitations.panel_text():
        with st.expander(section["heading"]):
            st.write(section["body"])

    with st.expander("What do all these numbers mean?"):
        number_glossary()


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
        "Disclaimer. Coursework only. Predictions describe patterns in that data and are not advice about "
        "anyone's earnings. "
        "Using it for a real decision would risk indirect discrimination under the Equality Act 2010."
    )


if __name__ == "__main__":
    main()