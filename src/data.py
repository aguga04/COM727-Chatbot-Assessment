"""Loading and feature preparation for the UCI Adult census dataset.

The single definition of how a raw census record becomes model input. Imported
by the training script, the inference path and the analysis notebook, so that
all three apply identical transformations.

Fitting returns a schema dictionary recording the retained country categories,
the levels of each categorical field and the final column order. Passing that
schema back reapplies those decisions to later data, including a single
unlabelled record entered through the application.
"""

import pandas as pd

from src import config


def load_raw():
    """Read the training and held-out test files.

    Returns a tuple of two DataFrames with the original fifteen columns.
    Question marks are read as missing values, and the trailing full stop that
    the test file appends to every income label is removed.
    """
    train = pd.read_csv(
        config.DATA_RAW / "adult.data",
        names=config.COLS,
        skipinitialspace=True,
        na_values="?",
    )
    test = pd.read_csv(
        config.DATA_RAW / "adult.test",
        names=config.COLS,
        skipinitialspace=True,
        na_values="?",
        skiprows=1,
    )
    test[config.TARGET] = test[config.TARGET].str.rstrip(".")
    return train, test


def extract_target(df_raw):
    """Return the binary target as a Series of zeros and ones.

    A one indicates that the record falls in the upper income bracket.
    """
    return (df_raw[config.TARGET] == config.POSITIVE_LABEL).astype(int)


def _group_rare_countries(series, country_categories):
    """Replace any country outside the retained set with a single label."""
    return series.where(series.isin(country_categories), config.OTHER_COUNTRY_LABEL)


def prepare_features(df_raw, schema=None):
    """Convert raw records into the numeric feature matrix the models expect.

    With ``schema=None`` the preparation is fitted on the supplied data and the
    resulting schema returned. With a schema supplied, those already fitted
    decisions are reapplied unchanged.

    Returns the prepared DataFrame and the schema used.
    """
    df = df_raw.drop(columns=config.DROP_COLS, errors="ignore").copy()
    df = df.drop(columns=[config.TARGET], errors="ignore")

    for column in config.MISSING_FILL_COLS:
        df[column] = df[column].fillna(config.MISSING_FILL_VALUE)

    fitting = schema is None
    if fitting:
        schema = {}
        frequencies = df["native-country"].value_counts(normalize=True)
        schema["country_categories"] = frequencies[
            frequencies >= config.RARE_COUNTRY_THRESHOLD
        ].index.tolist()

    df["native-country"] = _group_rare_countries(
        df["native-country"], schema["country_categories"]
    )

    if fitting:
        schema["category_levels"] = {
            column: sorted(df[column].dropna().unique().tolist())
            for column in config.CATEGORICAL_COLS
        }

    for column in config.CATEGORICAL_COLS:
        df[column] = pd.Categorical(
            df[column], categories=schema["category_levels"][column]
        )

    df = pd.get_dummies(df, columns=config.CATEGORICAL_COLS, drop_first=True)
    df = df.astype(float)

    if fitting:
        schema["feature_columns"] = df.columns.tolist()
    else:
        df = df.reindex(columns=schema["feature_columns"], fill_value=0.0)

    return df, schema


def compute_defaults(df_raw):
    """Derive fallback values for unanswered fields.

    Median for numeric fields, mode for categorical, computed on the training
    rows only. Returned as a dictionary keyed by column name.
    """
    df = df_raw.drop(columns=config.DROP_COLS, errors="ignore")
    defaults = {}

    for column in config.NUMERIC_COLS:
        defaults[column] = float(df[column].median())

    for column in config.CATEGORICAL_COLS:
        filled = df[column].fillna(config.MISSING_FILL_VALUE)
        defaults[column] = filled.mode().iloc[0]

    return defaults


def build_person(answers, defaults):
    """Assemble one record from partial input.

    Fields absent from ``answers`` are taken from ``defaults``. Returns a
    single-row DataFrame in the raw column layout and the sorted list of fields
    that took a default.
    """
    record = dict(defaults)
    supplied = {k: v for k, v in answers.items() if v is not None}
    record.update(supplied)

    defaulted = sorted(set(defaults) - set(supplied))
    return pd.DataFrame([record]), defaulted


if __name__ == "__main__":
    train, test = load_raw()
    print(f"train {train.shape}  test {test.shape}")

    X_train, schema = prepare_features(train)
    X_test, _ = prepare_features(test, schema)
    print(f"features {X_train.shape[1]}  aligned {list(X_train.columns) == list(X_test.columns)}")

    y_train = extract_target(train)
    print(f"upper bracket share {y_train.mean():.4f}")

    defaults = compute_defaults(train)
    person, defaulted = build_person({"age": 38, "education": "Bachelors"}, defaults)
    X_person, _ = prepare_features(person, schema)
    print(f"single record {X_person.shape}  defaulted {len(defaulted)} fields")
