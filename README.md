# Nexus Income Bracket Chatbot

A decision support helper for income bracket assessment. It classifies whether a person falls above or
below the US census income threshold, explains in plain language which attributes drove that outcome,
and refers cases it cannot separate to a human assessor rather than deciding them automatically.

**Disclaimer:** The classifier learns from the 1994 US Census. That vintage is a limitation the
application states plainly rather than works around, and examining what it does to the outcomes is
part of what the project sets out to demonstrate. This project was built as coursework and must not be
used to assess a real person. Using a system of this kind for a lending, employment or housing
decision would risk indirect discrimination under the Equality Act 2010.

Group project by [Team Nexus](#team-nexus-members) for COM727 Introduction to AI, MSc Applied AI and
Data Science, Southampton Solent University.

**Live application:** _pending deployment_

---

## Contents

- [The task](#the-task)
- [What the application does](#what-the-application-does)
- [Architecture](#architecture)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Reproducing the results](#reproducing-the-results)
- [Running the application](#running-the-application)
- [Results](#results)
- [The optional question feature](#the-optional-question-feature)
- [Limitations](#limitations)
- [Team Nexus Members](#team-nexus-members)
- [References](#references)
- [Licence](#licence)

---

## The task

**Binary classification.** Given the demographic and employment attributes recorded for one person by
the 1994 US Census, predict which of two income brackets they fall into.

| | |
| --- | --- |
| One input sample | Thirteen attributes: age, education, years of education, occupation, employer type, hours worked per week, marital status, household role, sex, race, country of birth, capital gains, capital losses |
| Output class 1 | **Upper bracket**, annual income over $50,000 |
| Output class 2 | **Lower bracket**, annual income of $50,000 or under |
| Selected classifier | **XGBoost**, chosen from eight candidates on validation F1 |
| Interface | The **Assessment tab** of the Streamlit application, which sends a new sample to the saved classifier |

**How the data is used.**

| Split | Records | Role |
| --- | ---: | --- |
| Training | 26,048 | The eight candidate models are fitted on these rows only |
| Validation | 6,513 | Model selection, decision band boundaries, calibration check |
| Official test (`adult.test`) | 16,281 | Untouched until every choice was final, then used once |

Model selection used validation data. Final performance used the official test file, which played no
part in choosing the algorithm, the hyperparameters or the band boundaries.

All preparation rules, meaning the rare-country grouping, the one-hot category levels and the scaling
values, are fitted on the training rows only and then applied unchanged to validation and test.

---

## What the application does

Enter a description of a person on the Assessment tab. The application returns three things.

**A predicted bracket.** The saved XGBoost model returns the probability that the person is in the
upper bracket. Above 50 percent the answer is the upper bracket; below it, the lower bracket.

**A decision band.** That probability is mapped onto one of three bands: confidently below the
threshold, confidently above it, or an uncertain middle band where the case is escalated to a human
rather than decided automatically. This is the accept, refer, decline pattern used in credit
screening. The boundaries were selected from validation evidence and then checked once on the official
test file.

**An explanation.** Every attribute's exact contribution to the prediction, measured in log odds and
translated into plain language. The contributions and the model's baseline sum to the raw output, and
converting that sum reproduces the predicted probability precisely, so the explanation can be checked
rather than trusted.

The application also carries an optional question and answer feature, described
[below](#the-optional-question-feature). It plays no part in predicting income.

---

## Architecture

Training and inference are separate. Model fitting happens offline in `src/train.py`, which writes
artefacts to `models/`. The application loads those artefacts and never fits a model, so a deployed
instance starts quickly and always reproduces the figures reported here.

```
adult.data  ──►  split  ──►  fit preparation on training rows  ──►  fit XGBoost
                                          │                              │
                                          ▼                              ▼
                             models/preprocessing.joblib     models/income_model.json
                                          │                              │
                                          └────────────┬─────────────────┘
                                                       ▼
                              Streamlit Assessment tab: new sample ──► prediction
```

No result is written into application code. Every figure the interface states is read from a generated
file at the moment it is displayed, so retraining changes what the application says.

---

## Repository layout

```
streamlit_app.py              Application entry point
pytest.ini                    Test configuration, puts the repository root on the import path
requirements.txt              Runtime dependencies, installed on the deployment server
requirements-dev.txt          Adds notebook and testing tools for local work

data/
  raw/                        Original UCI Adult files
  intents.json                Question set for the optional feature

notebooks/
  01_analysis_and_modeling.ipynb    EDA, eight model comparison, evaluation,
                                    calibration, band selection

src/
  config.py                   Paths, column groupings, constants, random seed
  data.py                     Loading and feature preparation, shared with the notebook
  train.py                    Fits the classifier and writes every artefact
  predict.py                  Loads artefacts and scores one new sample
  explain.py                  TreeSHAP contributions grouped by attribute
  decision.py                 Probability to decision band
  fairness.py                 Counterfactual, removal and proxy recovery audit
  nlp.py                      Text preprocessing, optional feature only
  chatbot/
    limitations.py            Standing limitations text, composed from generated figures
    engine.py                 Question routing, optional feature only

models/                       Generated artefacts and result files, committed
outputs/                      Generated figures, not committed
tests/
  test_pipeline.py            36 checks
```

---

## Installation

Requires **Python 3.12 or later**. Developed on 3.13. Use the same version you intend to deploy,
because the deployed Python version cannot be changed without recreating the application.

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

### Language resources, needed only by the optional feature

The optional question and answer feature uses NLTK, which does not ship its corpora with the package.
Three are downloaded on first use: `wordnet`, `omw-1.4` and `stopwords`. This happens automatically
inside a cached function, takes a few seconds and requires network access.

**The required classifier does not depend on this.** If the download is blocked, the Assessment, Why
and Model tabs work normally and the optional tab reports that it is unavailable. To fetch the corpora
in advance, or to test whether your network permits it:

```bash
python -c "import nltk; print(nltk.download('wordnet'), nltk.download('omw-1.4'), nltk.download('stopwords'))"
```

Three `True` values means the resources are available.

---

## Reproducing the results

Everything is deterministic: the random seed is fixed in `src/config.py` and the models are fitted on
the same split every time.

### 1. Train

```bash
python -m src.train
```

Expected output:

```
Income classifier
  train 26048  validation 6513  held-out test 16281  features 61
  split             accuracy  precision   recall       f1   roc_auc
  training            0.8903     0.8207   0.6966   0.7536    0.9482
  validation          0.8747     0.7789   0.6696   0.7202    0.9314
  held_out_test       0.8723     0.7696   0.6557   0.7081    0.9273
  held-out test: 2522 upper-bracket found, 1324 missed, 755 wrongly flagged,
                 11680 lower-bracket correct
```

Writes `income_model.json`, `preprocessing.joblib` and `metadata.json`, and also fits the optional
question classifier to `intent_model.joblib`.

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
the exploratory analysis, the eight model comparison with confusion matrices, the training against
validation against test check, the calibration measurement, the decision band selection and its
verification on the official test file.

It writes `model_comparison.csv`, `val_vs_test_all_models.csv`, `train_val_test_all_models.csv`,
`decision_thresholds.json` and `intent_metrics.json`, along with the figures in `outputs/`.

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

Opens at `http://localhost:8501`. Run it from the repository root; Streamlit resolves paths relative to
the root both locally and when deployed. The committed artefacts mean the application runs without
retraining.

**Assessment** is the required classifier interface. Enter a new sample and it returns the predicted
bracket, the probability, the decision band and the recommended action. Every field is a dropdown
restricted to values present in the training data, or a number input bounded by the range observed
there, so an impossible sample cannot be submitted. Fields whose value matches the training default
are listed with the result, because a default is an assumption rather than a blank.

**Why** shows the contribution of each attribute as a diverging bar chart with a plain language
translation, and a collapsed technical panel demonstrating that the contributions reconstruct the
model output exactly.

**Ask (optional)** is described below.

**Model and limitations** reports the eight algorithm comparison, the band selection evidence, the
fairness audit and the limitations in full.

---

## Results

Metrics are written by the scripts that compute them and are not transcribed anywhere by hand. Every
figure states which split it measures.

| File | Contains |
| --- | --- |
| `models/metadata.json` | Training, validation and test metrics, confusion matrix, hyperparameters, row counts, library versions |
| `models/model_comparison.csv` | Validation metrics for all eight algorithms |
| `models/val_vs_test_all_models.csv` | Validation against official test for all eight |
| `models/train_val_test_all_models.csv` | All three splits for all eight, with the train minus test gap |
| `models/decision_thresholds.json` | Band boundaries, calibration, and band results on both validation and official test |
| `models/fairness_report.json` | Group rates, counterfactuals, removal test, proxy recovery |
| `models/intent_metrics.json` | Optional feature only, reported separately |

### The selected classifier

| Split | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training | 0.8903 | 0.8207 | 0.6966 | 0.7536 | 0.9482 |
| Validation | 0.8747 | 0.7789 | 0.6696 | 0.7202 | 0.9314 |
| Official test | 0.8723 | 0.7696 | 0.6557 | 0.7081 | 0.9273 |

The gap between training and official test accuracy is 1.8 percentage points, so overfitting is
limited.

### Reading the errors

On the official test file the model finds 2,522 of the 3,846 people genuinely in the upper bracket and
misses 1,324 of them, and it wrongly flags 755 people as upper bracket. That is recall of 65.6
percent: **about one in three qualifying people is missed**, despite overall accuracy of 87.2 percent.

Accuracy flatters the model because the classes are imbalanced. Only 24.1 percent of records are in
the upper bracket, so predicting the lower bracket for everybody would score 75.9 percent without
learning anything. Recall is lower than precision because the upper bracket is the minority class, the
default 0.5 threshold favours the majority on an imbalanced problem, and the two groups genuinely
overlap in this data. The decision band is the direct response to that overlap.

### The decision bands

| Measure | Validation | Official test |
| --- | ---: | ---: |
| Referral rate | 17.80% | 17.72% |
| Accuracy on automated cases | 93.59% | 92.95% |
| Actual upper-bracket rate inside the band | 47.63% | 46.76% |

The boundaries of 0.30 and 0.70 were selected on validation and applied once to the official test
file. The near identical figures indicate they describe a property of the model rather than of the
split they were tuned on. The Brier score is 0.0863 and the expected calibration error 0.0061, both
measured on validation, so the probabilities can be read as probabilities rather than only as a
ranking.

---

## The optional question feature

The **Ask (optional)** tab is an additional help feature, not the required classifier interface. It
does not predict income and can be removed without affecting the classifier or the Assessment tab.

It offers thirty eight questions as buttons grouped by topic. Rather than mapping each button directly
to stored text, the button sends its question wording to a separate Multinomial Naive Bayes classifier
trained on 305 hand written phrasings, which recognises the intent and composes an answer. Answers
about the current assessment are assembled from live model output, so changing the person changes both
the wording and the figures. Every answer carries a panel showing which intent was recognised and with
what confidence.

Its results are reported separately from the classifier results and are not mixed with them:

| Measure | Value |
| --- | ---: |
| Accuracy on the thirty eight button wordings, held out of training | 100% |
| Cross validated accuracy across all 305 authored phrasings | 38.4% |
| Macro F1 across thirty eight intents | 0.380 |

The two figures measure different things. The first covers the wordings the interface actually
produces, none of which appeared in training. The second is far lower because thirty eight closely
related intents with eight examples each is a hard problem. Free text input is not offered, which is
one reason why. Full detail is in `models/intent_metrics.json`.

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

**The model is wrong regularly.** It misses about one in three people who genuinely are in the upper
bracket, as set out above.

**Outcomes differ by sex.** On the official test file the model predicts the upper bracket for 26.0
percent of men and 8.4 percent of women, where the actual rates in the data are 30.0 and 10.9 percent.
Among people who genuinely are above the threshold it identifies 66.5 percent of men and 60.3 percent
of women, so the disparity appears in error as well as in outcome.

**Removing protected attributes does not fix it.** Retraining without sex, race and country of birth
closes about nine percent of the gap while barely changing accuracy, because other columns carry the
same information. A classifier recovers sex from the remaining features at 84.6 percent, and still at
79.7 percent once the household role column is removed as well.

These are observed differences that warrant investigation. They are evidence of disparate outcome, and
they identify a mechanism, but they do not by themselves establish every cause, and nothing here is a
finding of unlawful discrimination.

**It must not be used to decide about real people.** Applying a system of this kind to lending,
employment or housing would raise indirect discrimination questions under sections 19 and 29 of the
Equality Act 2010, and the evidence above indicates it would not withstand that scrutiny.

---

## Team Nexus Members

- Ausbeth Aguguo
- Bhoomika Sri Bollu
- Darya Huryniuk
- Hillary Okojie
- Folakemi Olafisoye

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

Source code released under the GNU General Public Licence, as recorded in `LICENSE`.

The UCI Adult dataset is distributed by the UCI Machine Learning Repository under a Creative Commons
Attribution 4.0 International licence.