# Income Bracket Decision Support Chatbot

An interactive web application that predicts whether a person falls above or below the US Census income
threshold, explains which factors drove that prediction, and answers questions about the model through a
guided conversational interface.

Built for COM727 Machine Learning and Data Science, Southampton Solent University.

**Live application:** _pending deployment_

## Overview

The application takes a description of a person, drawn from the fields available in the UCI Adult census
dataset, and returns three things. First, a predicted probability that the person falls in the upper income
bracket. Second, a decision band derived from that probability, which routes confident cases to an automated
outcome and uncertain cases to human review. Third, a breakdown of how much each attribute contributed to the
prediction.

A conversational layer sits alongside the assessment. Users select from a set of suggested questions covering
the prediction, the model, the underlying data and the ethical limitations of the system. Each question is
classified into an intent by a trained natural language model, and answers about the current assessment are
composed from live model output rather than selected from stored text.

The application is a demonstration of a decision support pattern. It is not a lending tool and must not be
used to make decisions about real people. The limitations section below explains why.

## Dataset

UCI Adult, also known as Census Income. 48,842 records drawn from the 1994 United States Census, with 14
attributes covering demographics, education, employment and capital income. The prediction target is binary:
whether annual income exceeds fifty thousand US dollars.

Becker, B. and Kohavi, R. (1996). Adult. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20

The raw files are included under `data/raw/` so that the project can be reproduced without an external
download.

## Architecture

The system contains two trained models serving different purposes.

**Intent classifier.** A Multinomial Naive Bayes model trained on hand authored question patterns held in
`data/intents.json`. Input text is tokenised, lowercased, stripped of punctuation and stopwords, lemmatised
and converted to a bag of words vector. The model returns a probability distribution across the intent set,
and a confidence threshold determines whether the system answers or returns a fallback.

**Income classifier.** An XGBoost model trained on the census data, selected on validation performance against
seven alternatives. Per attribute contributions are obtained through the TreeSHAP implementation built into
XGBoost, which produces an additive decomposition: the contributions and the bias term sum to the raw margin,
and the sigmoid of that sum reproduces the predicted probability exactly.

Training and inference are separated. Model fitting happens offline in `src/train.py`, which writes artefacts
to `models/`. The application loads those artefacts and never fits a model at runtime.

## Repository structure

```
streamlit_app.py         Application entry point
data/
  intents.json           Conversational training data and interface question set
  raw/                   Original UCI Adult files
notebooks/               Exploratory analysis and model comparison
src/
  config.py              Paths and shared constants
  data.py                Loading and feature preparation
  train.py               Model fitting and artefact production
  predict.py             Single record inference
  explain.py             Per attribute contribution decomposition
  decision.py            Probability to decision band mapping
  nlp.py                 Text preprocessing for the intent classifier
  fairness.py            Counterfactual and proxy recovery audit
  chatbot/               Intent definitions and response engine
models/                  Trained artefacts, metadata and generated reports
tests/                   Automated checks
```

## Installation

Requires Python 3.12 or later.

```
git clone <repository-url>
cd COM727-Chatbot-Assessment
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
```

On macOS or Linux, activate with `source .venv/bin/activate`.

`requirements.txt` holds the runtime dependencies installed on the deployment server.
`requirements-dev.txt` adds the notebook and testing tools needed for local development.

## Usage

Train the models and produce the artefacts:

```
python -m src.train
```

Generate the fairness audit:

```
python -m src.fairness
```

Run the application:

```
streamlit run streamlit_app.py
```

The application opens at `http://localhost:8501`. All commands are run from the repository root, because
Streamlit resolves paths relative to the root both locally and when deployed.

Run the test suite:

```
pytest -q
```

## Results

Evaluation metrics are recorded in `models/metadata.json` for the income classifier and
`models/intent_metrics.json` for the intent classifier, and are written by the training scripts rather than
transcribed by hand. The full eight algorithm comparison is in `models/model_comparison.csv`. All three are
displayed inside the application.

Reported figures cover held out test performance for the income classifier, per intent precision, recall and
F1 with a confusion matrix for the intent classifier, and the calibration assessment used to set the decision
band boundaries.

## Limitations

The training data is from 1994. The fifty thousand dollar threshold that defines the target reflects the
income distribution of that year and no longer corresponds to a meaningful boundary, a problem documented in
detail by Ding et al. (2021). The data describes the United States and carries no direct relevance to any
other economy.

Removing sex, race and country of birth from the feature set does not produce a model that treats groups
alike. Other attributes act as proxies, most directly the household role field, which distinguishes husband
from wife. The audit in `src/fairness.py` quantifies this, along with the effect of altering the sex field on
otherwise identical records. Findings are written to `models/fairness_report.json` and surfaced in the
application.

The decision bands are derived from predicted probability, not from any measurement of income. A prediction in
the upper band states that the model expects the person to fall above the threshold. It is not a statement
about what anyone earns.

Applying a system of this kind to lending, employment or any similar decision would raise indirect
discrimination questions under the Equality Act 2010 and would not be defensible on this data. The referral
band exists so that cases the model cannot separate are escalated to a person rather than decided
automatically.

## Authors

Group project, COM727, MSc Applied AI and Data Science, Southampton Solent University.

## Licence

Source code is released under the MIT Licence, as recorded in `LICENSE`. The UCI Adult dataset is distributed
by the UCI Machine Learning Repository under a Creative Commons Attribution 4.0 International licence.