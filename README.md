# Income Bracket Decision Support Chatbot

A web application that predicts whether a person falls above or below the 1994 US census income
threshold, explains exactly which attributes drove that prediction, routes uncertain cases to a human
assessor, and answers questions about itself through a trained conversational model.

Built for COM727 Introduction to AI, Southampton Solent University.

**Live application:** _pending deployment_

---

## Contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Reproducing the results](#reproducing-the-results)
- [Running the application](#running-the-application)
- [Results](#results)
- [Deployment](#deployment)
- [Limitations](#limitations)
- [References](#references)
- [Licence](#licence)

---

## What it does

Enter a description of a person using the fields available in the UCI Adult census dataset. The
application returns three things.

**A decision band.** The predicted probability of the upper income bracket is mapped onto one of three
bands: confidently below the threshold, confidently above it, or an uncertain middle band where the
case is escalated to a human rather than decided automatically. This is the accept, refer, decline
pattern used in credit screening. The boundaries were selected from validation evidence, not chosen
for neatness.

**An explanation.** Every attribute's exact contribution to the prediction, measured in log odds. The
contributions and the model's baseline sum to the raw output, and converting that sum reproduces the
predicted probability precisely, so the explanation can be checked rather than trusted.

**Answers.** A separate trained classifier recognises which of thirty eight questions was asked and
composes a reply. Answers about the current assessment are assembled from live model output, so
changing the person changes both the wording and the figures.

The application is a demonstration of a decision support pattern. It is not a lending tool. See
[Limitations](#limitations).

---

## Architecture

Two trained models do different jobs.

| | Income classifier | Intent classifier |
| --- | --- | --- |
| Algorithm | XGBoost | Multinomial Naive Bayes |
| Trained on | 26,048 census records | 305 authored question phrasings |
| Predicts | Which side of the income threshold | Which of 38 questions was asked |
| Explained by | TreeSHAP contributions | Confidence with a fallback threshold |
| Artefacts | `income_model.json`, `preprocessing.joblib` | `intent_model.joblib` |

Training and inference are separate. Model fitting happens offline in `src/train.py`, which writes
artefacts to `models/`. The application loads those artefacts and never fits a model, so a deployed
instance starts quickly and always reproduces the figures reported here.

No result is written into application code or into the authored responses. Every figure the interface
states is read from a generated file at the moment it is displayed, so retraining changes what the
application says.

---

## Repository layout

```
streamlit_app.py              Application entry point
pytest.ini                    Test configuration, puts the repository root on the import path
requirements.txt              Runtime dependencies, installed on the deployment server
requirements-dev.txt          Adds notebook and testing tools for local work

data/
  intents.json                Conversational training data and the interface question set
  raw/                        Original UCI Adult files

notebooks/
  01_analysis_and_modeling.ipynb    Exploration, eight model comparison, calibration,
                                    band selection, intent evaluation

src/
  config.py                   Paths, column groupings, constants
  data.py                     Loading and feature preparation, shared with the notebook
  train.py                    Fits both models and writes every artefact
  predict.py                  Loads artefacts and scores one person
  explain.py                  TreeSHAP contributions grouped by attribute
  decision.py                 Probability to decision band
  nlp.py                      Tokenise, lemmatise, bag of words
  fairness.py                 Counterfactual, removal and proxy recovery audit
  chatbot/
    limitations.py            Standing limitations text, composed from generated figures
    engine.py                 Intent routing and response composition

models/                       Generated artefacts and result files, committed
outputs/                      Generated figures, not committed
tests/
  test_pipeline.py            36 checks
```

---

## Installation

Requires **Python 3.12 or later**. Develop on the same version you intend to deploy, because the
deployed Python version cannot be changed without recreating the application.

```bash
git clone <repository-url>
cd COM727-Chatbot-Assessment

python -m venv .venv
.venv\Scripts\activate          # macOS or Linux: source .venv/bin/activate

pip install -r requirements-dev.txt
```

`requirements.txt` holds the runtime dependencies. `requirements-dev.txt` installs those and adds
Jupyter, matplotlib, seaborn and pytest.

Confirm the configuration resolves before going further:

```bash
python -c "from src import config; print(config.ROOT); print(config.DATA_RAW.exists())"
```

This should print the repository path followed by `True`. If it raises a module error, you are not
running from the repository root.

**Modules under `src/` are run as modules, never as file paths.** Use `python -m src.train`, not
`python src/train.py`. Running a file directly puts that file's own folder on the import path instead
of the repository root, and the imports fail.

---

## Reproducing the results

Three commands, in order. Everything is deterministic: the random seed is fixed in `src/config.py`
and both models are fitted on the same split every time.

### 1. Train both models

```bash
python -m src.train
```

Expected output:

```
Income classifier
  train 26048  validation 6513  held-out test 16281  features 61
  split             accuracy  precision   recall       f1   roc_auc
  validation          0.8747     0.7789   0.6696   0.7202    0.9314
  held_out_test       0.8723     0.7696   0.6557   0.7081    0.9273

Intent classifier
  38 intents, 305 patterns, 323 words after preprocessing
  alpha 0.3  binary presence True
  display strings resolving correctly 100.0%
  confidence on buttons: lowest 0.116, median 0.830
```

Writes `income_model.json`, `preprocessing.joblib`, `metadata.json` and `intent_model.joblib`.

If the numbers differ, something upstream has changed. Check the seed, the split proportion and the
hyperparameters in `src/train.py` before continuing.

### 2. Run the fairness audit

```bash
python -m src.fairness
```

Fits four additional models, so allow a minute or two. Writes `fairness_report.json`.

Expected findings: reversing the sex field on the held-out file changes 544 predictions; deleting sex,
race and country of birth closes about nine percent of the selection rate gap; sex remains recoverable
from the other columns at 84.6 percent against a 66.9 percent baseline.

### 3. Run the notebook

Open `notebooks/01_analysis_and_modeling.ipynb`, restart the kernel and run all cells. This produces
the eight model comparison, the calibration check, the decision band selection and the intent
evaluation, writing `model_comparison.csv`, `val_vs_test_all_models.csv`, `decision_thresholds.json`
and `intent_metrics.json`, along with the figures in `outputs/`.

The notebook imports its data loading and preparation from `src.data`, so the notebook and the
application encode features identically by construction rather than by agreement.

### 4. Verify

```bash
pytest -q
```

Expected: `36 passed`.

The suite checks the things that fail quietly rather than loudly. Two are regression tests for defects
found during development: a single record whose categorical encoding silently collapsed, and a
question whose button label had leaked into its own training data. It also asserts that the held-out
accuracy computed now matches the figure recorded in `metadata.json`, so a stale artefact cannot pass
unnoticed.

---

## Running the application

```bash
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`. Run it from the repository root; Streamlit resolves paths relative
to the root both locally and when deployed.

Four tabs:

**Assessment** enters a person and returns the band, the probability and the recommended action. Every
field is a dropdown restricted to values present in the training data, or a number input bounded by
the range observed there. Fields left at their training default are listed with the result, because a
default is an assumption rather than a blank.

**Why** shows the contribution of each attribute as a diverging bar chart, with the arithmetic
underneath demonstrating that the contributions reconstruct the model output exactly.

**Ask** offers the thirty eight questions as buttons grouped by topic. Each button sends its question
text through the intent classifier exactly as typed input would arrive, and every answer carries a
panel showing which intent was recognised and with what confidence.

**Model and limitations** reports the eight algorithm comparison, the band selection evidence, the
conversational model's own evaluation, the fairness audit, and the limitations in full.

On first run the application downloads three NLTK corpora. This happens once, inside a cached
function, and takes a few seconds.

---

## Results

Metrics are written by the scripts that compute them and are not transcribed anywhere by hand.

| File | Contains |
| --- | --- |
| `models/metadata.json` | Income model metrics, hyperparameters, row counts, library versions |
| `models/model_comparison.csv` | Validation metrics for all eight algorithms |
| `models/val_vs_test_all_models.csv` | Validation against held-out test for all eight |
| `models/decision_thresholds.json` | Band boundaries, referral rate, calibration, band outcome rates |
| `models/intent_metrics.json` | Per intent precision, recall and F1, confusion pairs, threshold |
| `models/fairness_report.json` | Group rates, counterfactuals, removal test, proxy recovery |

Headline figures, all reproducible from the commands above:

- Held-out accuracy 0.8723, F1 0.7081, ROC-AUC 0.9273 on 16,281 records never seen in training
- Referral band 0.30 to 0.70, referring 17.8 percent of cases and raising accuracy on the automated
  remainder from 0.8747 to 0.9359
- Brier score 0.0863 and expected calibration error 0.0061, so the probabilities can be read as
  probabilities
- Intent classifier: 100 percent on the thirty eight held-out button phrasings, 38.4 percent
  cross validated across all authored phrasings

The two intent figures measure different things and both are reported. The first covers the wordings
the interface actually produces, none of which appeared in training. The second is cross validated
over all 305 phrasings and is far lower, because thirty eight closely related intents with eight
examples each is a hard problem. Free text input is not offered, which is one reason why.

---

## Limitations

Read these before drawing any conclusion from a prediction. The application states them in full on its
own limitations tab, generated from the audit files rather than written by hand.

**The data is from 1994.** The fifty thousand dollar threshold has never been adjusted, so it no longer
marks the same position in the income distribution. Ding et al. (2021) argue the dataset should be
retired from fairness research for this reason.

**A band is not an income estimate.** The dataset contains no continuous income figure. The model
predicts which side of a threshold a person falls on, and the bands describe confidence in that
prediction rather than income tiers.

**Outcomes differ by sex.** On the held-out file the model predicts the upper bracket for 26.0 percent
of men and 8.4 percent of women. Among people who genuinely are above the threshold it identifies 66.5
percent of men and 60.3 percent of women, so the disparity is in error as well as in outcome.

**Removing protected attributes does not fix it.** Retraining without sex, race and country of birth
closes about nine percent of the gap while barely changing accuracy, because other columns carry the
same information. A classifier recovers sex from the remaining features at 84.6 percent, and still at
79.7 percent once the household role column is removed as well.

**It must not be used to decide about real people.** Applying a system of this kind to lending,
employment or housing would raise indirect discrimination questions under sections 19 and 29 of the
Equality Act 2010, and the evidence above indicates it would not withstand that scrutiny.

---

## References

Becker, B. and Kohavi, R. (1996) *Adult*. UCI Machine Learning Repository.
https://doi.org/10.24432/C5XW20

Barocas, S. and Selbst, A. (2016) Big Data's Disparate Impact. *California Law Review*, 104, 671–732.

Chen, T. and Guestrin, C. (2016) XGBoost: A Scalable Tree Boosting System. *Proceedings of KDD 2016*,
785–794.

Ding, F., Hardt, M., Miller, J. and Schmidt, L. (2021) Retiring Adult: New Datasets for Fair Machine
Learning. *Advances in Neural Information Processing Systems 34*.

Lundberg, S. et al. (2020) From local explanations to global understanding with explainable AI for
trees. *Nature Machine Intelligence*, 2(1), 56–67.

Manning, C., Raghavan, P. and Schütze, H. (2008) *Introduction to Information Retrieval*. Cambridge
University Press.

Niculescu-Mizil, A. and Caruana, R. (2005) Predicting good probabilities with supervised learning.
*Proceedings of ICML 2005*.

Rudin, C. (2019) Stop explaining black box machine learning models for high stakes decisions and use
interpretable models instead. *Nature Machine Intelligence*, 1(5), 206–215.

Thomas, L., Edelman, D. and Crook, J. (2002) *Credit Scoring and Its Applications*. SIAM.

---

## Licence

Source code released under the MIT Licence, as recorded in `LICENSE`.

The UCI Adult dataset is distributed by the UCI Machine Learning Repository under a Creative Commons
Attribution 4.0 International licence.