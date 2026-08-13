"""Project-wide paths and constants.

Every module resolves its file locations through this file so that the project
runs identically from any working directory and on any machine.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA_RAW = DATA / "raw"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"

RANDOM_STATE = 42

COLS = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income",
]

TARGET = "income"
POSITIVE_LABEL = ">50K"

DROP_COLS = ["fnlwgt"]

NUMERIC_COLS = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]

CATEGORICAL_COLS = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country",
]

MISSING_FILL_COLS = ["workclass", "occupation", "native-country"]
MISSING_FILL_VALUE = "Unknown"

RARE_COUNTRY_THRESHOLD = 0.01
OTHER_COUNTRY_LABEL = "Other"

# Education level and years of education are two encodings of the same
# attribute and correspond one to one throughout the dataset. The mapping is
# recorded here so the interface can ask for the level only and derive the
# numeric form, rather than defaulting a field the user has effectively
# already answered. Verify with:
#   train.groupby("education")["education-num"].nunique().max() == 1
EDUCATION_NUM = {
    "Preschool": 1,
    "1st-4th": 2,
    "5th-6th": 3,
    "7th-8th": 4,
    "9th": 5,
    "10th": 6,
    "11th": 7,
    "12th": 8,
    "HS-grad": 9,
    "Some-college": 10,
    "Assoc-voc": 11,
    "Assoc-acdm": 12,
    "Bachelors": 13,
    "Masters": 14,
    "Prof-school": 15,
    "Doctorate": 16,
}

# Ranges accepted by the assessment form, taken from the training data so that
# no one can enter a value the model has never seen. Verify with:
#   train[NUMERIC_COLS].agg(["min", "max"])
NUMERIC_RANGES = {
    "age": (17, 90),
    "hours-per-week": (1, 99),
    "capital-gain": (0, 99999),
    "capital-loss": (0, 4356),
}

# Fields the form asks for directly, and those tucked into the optional
# section. Years of education is absent from both because it is derived from
# the qualification rather than asked for.
PRIMARY_FIELDS = [
    "age", "education", "occupation", "workclass",
    "hours-per-week", "marital-status", "relationship", "sex",
]

OPTIONAL_FIELDS = ["capital-gain", "capital-loss", "race", "native-country"]

FIELD_LABELS = {
    "age": "Age",
    "education": "Highest qualification",
    "occupation": "Occupation",
    "workclass": "Employer type",
    "hours-per-week": "Hours worked per week",
    "marital-status": "Marital status",
    "relationship": "Household role",
    "sex": "Sex",
    "capital-gain": "Capital gains",
    "capital-loss": "Capital losses",
    "race": "Race",
    "native-country": "Country of birth",
}