"""Six metrics for binary or multiclass models (also reused by the Streamlit app)."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import label_binarize

METRIC_COLS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def get_scores(model, X):
    """Class scores for AUC: probabilities if available, otherwise decision scores."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


def safe_auc(y_true, scores, classes):
    """Binary AUC, or macro one-vs-rest AUC for multiclass. NaN if it cannot be computed."""
    if scores is None:
        return np.nan
    y_true = np.asarray(y_true)
    present = [i for i, c in enumerate(classes) if (y_true == c).any()]
    if len(present) < 2:                       # AUC needs at least two classes in y_true
        return np.nan
    try:
        if len(classes) == 2:
            s = scores if np.ndim(scores) == 1 else scores[:, 1]
            return roc_auc_score(y_true, s)
        Y = label_binarize(y_true, classes=classes)
        # one-vs-rest per class (works for probabilities and for raw decision scores);
        # classes absent from y_true are skipped
        aucs = [roc_auc_score(Y[:, i], scores[:, i]) for i in present]
        return float(np.mean(aucs))
    except ValueError:
        return np.nan


def compute_metrics(model, X, y_true_enc):
    """Return the six metrics (macro-averaged precision/recall/F1) plus predictions."""
    y_pred = model.predict(X)
    classes = np.asarray(model.classes_)
    scores = get_scores(model, X)
    row = {
        "Accuracy": accuracy_score(y_true_enc, y_pred),
        "AUC": safe_auc(y_true_enc, scores, classes),
        "Precision": precision_score(y_true_enc, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true_enc, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true_enc, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_true_enc, y_pred),
    }
    return row, y_pred


def evaluate_all(models, X_test, y_test_enc):
    """DataFrame in the assignment format: one row per model."""
    rows, preds = [], {}
    for name, model in models.items():
        r, p = compute_metrics(model, X_test, y_test_enc)
        rows.append({"ML Model": name, **r}); preds[name] = p
    return pd.DataFrame(rows).round(4), preds


def validate_test_frame(df, meta):
    """Check an uploaded CSV against the saved metadata.

    Returns (X, y_encoded, errors, notes). X is None when errors is non-empty.
    Nothing here is fitted on the uploaded data; it only checks and converts types.
    """
    errors, notes = [], []
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    if df.empty:
        return None, None, ["The file has no rows."], notes

    features, target = meta["feature_columns"], meta["target_column"]
    missing = [c for c in features + [target] if c not in df.columns]
    if missing:
        errors.append(f"Missing required column(s): {missing}")
        return None, None, errors, notes

    extra = [c for c in df.columns if c not in features + [target]]
    if extra:
        notes.append(f"Ignored {len(extra)} extra column(s): {extra}")

    X = df[features].apply(pd.to_numeric, errors="coerce")
    bad = [c for c in features if X[c].isna().sum() > df[c].isna().sum()]
    if bad:
        errors.append(f"Non-numeric values found in: {bad}")

    labels = df[target].astype(str).str.strip()
    unknown = sorted(set(labels.unique()) - set(meta["class_names"]))
    if unknown:
        errors.append(f"Unknown target label(s) {unknown}; expected only {meta['class_names']}")
    if errors:
        return None, None, errors, notes

    if int(X.isna().sum().sum()):
        notes.append("Blank cells were found; the saved pipeline fills them with values "
                     "learned from the training data.")
    if labels.nunique() < 2:
        notes.append("Only one class is present in the target column, so AUC cannot be computed (shown as N/A).")
    to_idx = {c: i for i, c in enumerate(meta["class_names"])}
    return X, labels.map(to_idx).to_numpy(), errors, notes