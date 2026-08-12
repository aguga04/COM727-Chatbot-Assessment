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