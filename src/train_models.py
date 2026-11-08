"""Build, tune (cross-validation on TRAINING data only) and save the five classifiers."""
import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from preprocessing import (BINARY_COLS, NOMINAL_COLS, RANDOM_STATE, TARGET_COL,
                           build_preprocessor)

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "SVM": "svm.joblib",
}
PREPROCESSOR_FILES = {
    name: filename.replace(".joblib", "_preprocessor.joblib")
    for name, filename in MODEL_FILES.items()
}


def model_zoo(numeric_cols):
    """name -> (full Pipeline, hyper-parameter grid). Grid keys start with 'clf__'."""
    scaled = lambda: build_preprocessor(numeric_cols, scale=True)
    unscaled = lambda: build_preprocessor(numeric_cols, scale=False)  # trees do not need scaling
    return {
        "Logistic Regression": (
            Pipeline([("prep", scaled()),
                      ("clf", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE))]),
            {"clf__C": [0.05, 0.1, 0.5, 1, 5]}),
        "Decision Tree": (
            Pipeline([("prep", unscaled()),
                      ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE))]),
            {"clf__max_depth": [4, 6, 8, 12], "clf__min_samples_leaf": [5, 10, 20]}),
        "KNN": (
            Pipeline([("prep", scaled()), ("clf", KNeighborsClassifier())]),
            {"clf__n_neighbors": [5, 9, 15, 25], "clf__weights": ["uniform", "distance"]}),
        "Naive Bayes": (
            Pipeline([("prep", scaled()), ("clf", GaussianNB())]),
            {"clf__var_smoothing": [1e-9, 1e-6, 1e-3, 1e-2, 1e-1]}),
        "SVM": (
            Pipeline([("prep", scaled()),
                      ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))]),
            {"clf__C": [0.5, 1, 5, 10], "clf__gamma": ["scale", 0.01]}),
    }


def train_all(X_train, y_train_enc, numeric_cols, cv_folds=5):
    """Grid-search each pipeline with stratified CV (macro-F1) on the training split only."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    fitted, rows = {}, []
    for name, (pipe, grid) in model_zoo(numeric_cols).items():
        search = GridSearchCV(pipe, grid, scoring="f1_macro", cv=cv, n_jobs=-1, refit=True)
        search.fit(X_train, y_train_enc)
        fitted[name] = search.best_estimator_
        rows.append({"ML Model": name,
                     "Best CV macro-F1": round(search.best_score_, 4),
                     "Best parameters": {k.replace("clf__", ""): v
                                         for k, v in search.best_params_.items()}})
    return fitted, pd.DataFrame(rows)


def save_all(fitted, label_enc, numeric_cols, X_columns, model_dir="../model"):
    """Write fitted classifiers, their preprocessors, and app metadata."""
    out = Path(model_dir); out.mkdir(parents=True, exist_ok=True)
    for name, pipeline in fitted.items():
        joblib.dump(pipeline.named_steps["clf"], out / MODEL_FILES[name])
        joblib.dump(pipeline.named_steps["prep"], out / PREPROCESSOR_FILES[name])
    meta = {
        "target_column": TARGET_COL,
        "class_names": [str(c) for c in label_enc.classes_],  # position = encoded label
        "feature_columns": list(X_columns),
        "numeric_columns": numeric_cols,
        "nominal_columns": NOMINAL_COLS,
        "binary_columns": BINARY_COLS,
        "model_files": MODEL_FILES,
        "preprocessor_files": PREPROCESSOR_FILES,
        "random_state": RANDOM_STATE,
        "sklearn_version": sklearn.__version__,
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    return meta