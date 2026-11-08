"""Shared preprocessing for the Student Dropout & Academic Success project."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COL = "Target"
RANDOM_STATE = 2026  # change to any number of your own choice, then keep it fixed

# Integer-coded labels with NO natural order -> one-hot encode
NOMINAL_COLS = [
    "Marital status", "Application mode", "Course", "Previous qualification",
    "Nacionality", "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation",
]
# Already 0/1 flags -> keep as they are
BINARY_COLS = [
    "Daytime/evening attendance", "Displaced", "Educational special needs",
    "Debtor", "Tuition fees up to date", "Gender", "Scholarship holder",
    "International",
]


def load_raw(path):
    """Read a comma- or semicolon-separated dataset and tidy column names."""
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()  # 'Daytime/evening attendance\t' has a hidden tab
    return df


def get_numeric_cols(df):
    return [c for c in df.columns if c not in NOMINAL_COLS + BINARY_COLS + [TARGET_COL]]


def build_preprocessor(numeric_cols, scale=True):
    """ColumnTransformer; fit it on TRAINING data only (inside a Pipeline)."""
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        transformers=[
            ("nominal", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                         min_frequency=0.01, sparse_output=False)),
            ]), NOMINAL_COLS),
            ("binary", SimpleImputer(strategy="most_frequent"), BINARY_COLS),
            ("numeric", Pipeline(num_steps), numeric_cols),
        ],
        remainder="drop",
    )