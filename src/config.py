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