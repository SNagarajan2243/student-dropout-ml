"""Student Success Radar - Streamlit front end for five saved classifiers.

The app only LOADS saved classifiers and fitted preprocessors from model/.
Nothing is ever trained or fitted on an uploaded file.
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from src.evaluate_models import METRIC_COLS, compute_metrics, validate_test_frame

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
SAMPLE_CSV = ROOT / "test_data.csv"

st.set_page_config(page_title="Student Success Radar", page_icon="🎓", layout="wide")

st.markdown(
    """
    <style>
    .hero {background: linear-gradient(120deg,#1f3c88 0%,#2a6f97 55%,#3aa6a0 100%);
           padding: 1.6rem 1.8rem; border-radius: 14px; color: #fff; margin-bottom: 1rem;}
    .hero h1 {margin: 0; font-size: 2rem; color: #fff;}
    .hero p {margin: .35rem 0 0 0; opacity: .92;}
    div[data-testid="stMetric"] {background: rgba(42,111,151,.09); border-radius: 12px;
           padding: .7rem .9rem; border: 1px solid rgba(42,111,151,.25);}
    </style>
    """,
    unsafe_allow_html=True,
)

ABOUT_MODELS = {
    "Logistic Regression": "Learns one weight per feature and turns a weighted sum into class probabilities (softmax). "
                           "Fast, stable and easy to interpret; works well when scaled features relate roughly linearly to the outcome.",
    "Decision Tree": "Asks a sequence of yes/no questions on feature values (for example, 1st-semester approved units) until it reaches a class. "
                     "Needs no scaling and is easy to read, but a single tree can overfit, so depth and leaf size are limited.",
    "KNN": "Predicts by majority vote among the k most similar students in the training set. "
           "Simple and flexible, but distance gets noisy in high-dimensional one-hot data and prediction is slower.",
    "Naive Bayes": "Applies Bayes' rule while assuming features are independent given the class (Gaussian likelihoods here). "
                   "Very quick and needs little data, but the independence assumption is rarely true for correlated academic features.",
    "SVM": "Finds the boundary with the widest margin between classes; the RBF kernel lets that boundary curve. "
           "Strong on scaled, medium-sized tabular data; C and gamma control how tightly it fits.",
}
PARAM_KEYS = {
    "Logistic Regression": ["C"], "Decision Tree": ["max_depth", "min_samples_leaf"],
    "KNN": ["n_neighbors", "weights"], "Naive Bayes": ["var_smoothing"], "SVM": ["kernel", "C", "gamma"],
}


# ---------------------------------------------------------------- loading (cached, never refitted)
@st.cache_resource(show_spinner="Loading saved models...")
def load_artifacts():
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    models = {
        name: Pipeline([
            ("prep", joblib.load(MODEL_DIR / meta["preprocessor_files"][name])),
            ("clf", joblib.load(MODEL_DIR / filename)),
        ])
        for name, filename in meta["model_files"].items()
    }
    return meta, models


@st.cache_data(show_spinner="Scoring all five models...")
def score_all(X, y_enc):
    _, models = load_artifacts()
    rows, preds = [], {}
    for name, model in models.items():
        m, p = compute_metrics(model, X, y_enc)
        rows.append({"ML Model": name, **m})
        preds[name] = p
    return pd.DataFrame(rows), preds


def read_csv_flexible(source):
    # sep=None lets pandas detect comma or semicolon; utf-8-sig removes a BOM if present
    return pd.read_csv(source, sep=None, engine="python", encoding="utf-8-sig")


def fmt(v):
    return "N/A" if pd.isna(v) else f"{v:.4f}"


try:
    meta, models = load_artifacts()
except Exception as exc:  # missing or incompatible model files
    st.error(f"Could not load the saved models from `model/`: {exc}")
    st.stop()

class_names = meta["class_names"]

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("1. Test data")
    uploaded = st.file_uploader("Upload a test CSV", type=["csv"],
                                help="Must contain every input feature column plus the 'Target' column.")
    use_sample = st.button("Use bundled test_data.csv", disabled=not SAMPLE_CSV.exists())
    st.header("2. Model")
    chosen = st.selectbox("Choose a classifier", list(models.keys()), index=4)
    st.header("3. Display")
    normalise = st.checkbox("Normalise confusion matrix (row %)", value=False)
    st.divider()
    st.caption(f"Models trained with scikit-learn {meta['sklearn_version']}; running {sklearn.__version__}.")
    if meta["sklearn_version"] != sklearn.__version__:
        st.warning("scikit-learn versions differ. Pin the training version in requirements.txt.")

if use_sample:
    st.session_state["use_sample"] = True
if uploaded is not None:
    st.session_state["use_sample"] = False

# ---------------------------------------------------------------- hero
st.markdown(
    """
    <div class="hero">
      <h1>🎓 Student Success Radar</h1>
      <p>Predict whether a student will <b>drop out</b>, stay <b>enrolled</b> or <b>graduate</b> —
         and compare five classic classifiers on your own held-out test file.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_about, tab_single, tab_compare = st.tabs(["📘 About", "🔍 Evaluate one model", "⚖️ Compare all models"])

with tab_about:
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Problem statement")
        st.write(
            "Universities lose students to dropout, and the earlier a risk is spotted the more can be done. "
            "Using information known at enrolment and the first two semesters, the task is a **three-class "
            "classification**: *Dropout*, *Enrolled* or *Graduate* at the end of the normal course duration."
        )
        st.subheader("Dataset")
        st.write(
            "**Predict Students' Dropout and Academic Success** (UCI Machine Learning Repository; Realinho et al., 2021): "
            "4,424 students enrolled between 2008 and 2019 in Portuguese higher education, with 36 input features "
            "(demographics, socio-economic factors, application details, 1st/2nd-semester academic results, "
            "macro-economic indicators) and one target column. Classes are imbalanced: about 50% Graduate, 32% Dropout, 18% Enrolled."
        )
        st.write(
            "An 80/20 stratified split was made; models were tuned by 5-fold cross-validation on the 80% training part only. "
            "This app works with the 20% hold-out file (885 students)."
        )
    with right:
        st.subheader("How to use")
        st.markdown(
            "1. Upload the test CSV (or press **Use bundled test_data.csv**)\n"
            "2. Pick a model in the sidebar\n"
            "3. Read the metrics, confusion matrix and report\n"
            "4. Open **Compare all models** for the side-by-side view"
        )
        st.info("Nothing is retrained: the app loads fitted pipelines and only calls `predict`.")
    st.subheader("The five algorithms")
    for name, text in ABOUT_MODELS.items():
        with st.expander(name):
            st.write(text)

# ---------------------------------------------------------------- data intake
X = y_enc = None
source_label = None
if uploaded is not None:
    source_label = uploaded.name
    try:
        raw = read_csv_flexible(uploaded)
    except Exception as exc:
        st.error(f"The file could not be read as a CSV: {exc}")
        raw = None
elif st.session_state.get("use_sample") and SAMPLE_CSV.exists():
    source_label = "bundled test_data.csv"
    raw = read_csv_flexible(SAMPLE_CSV)
else:
    raw = None

if raw is not None:
    X, y_enc, errors, notes = validate_test_frame(raw, meta)
    if errors:
        for e in errors:
            st.error(e)
        with st.expander("Columns this app expects"):
            st.write(meta["feature_columns"] + [meta["target_column"]])
        X = None
    else:
        for n in notes:
            st.sidebar.info(n)
        st.sidebar.success(f"Loaded {source_label}: {len(X):,} rows")

# ---------------------------------------------------------------- single model
with tab_single:
    if X is None:
        st.info("Upload a valid test CSV (or use the bundled one) from the sidebar to see results.")
    else:
        model = models[chosen]
        metrics, y_pred = compute_metrics(model, X, y_enc)
        st.subheader(f"{chosen} on {len(X):,} students")

        cols = st.columns(6)
        for col, key in zip(cols, METRIC_COLS):
            col.metric(key, fmt(metrics[key]))
        st.caption("Precision, Recall and F1 are macro-averaged; AUC is macro one-vs-rest.")

        params = {k: model.named_steps["clf"].get_params().get(k) for k in PARAM_KEYS[chosen]}
        with st.expander("Hyper-parameters stored in the saved model"):
            st.json({k: (v if isinstance(v, (int, float, str, type(None))) else str(v)) for k, v in params.items()})

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Confusion matrix**")
            cm = confusion_matrix(y_enc, y_pred, labels=range(len(class_names)))
            shown = cm / cm.sum(axis=1, keepdims=True).clip(min=1) * 100 if normalise else cm
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(shown, annot=True, fmt=".1f" if normalise else "d", cmap="YlGnBu",
                        xticklabels=class_names, yticklabels=class_names, ax=ax, cbar=False)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            st.pyplot(fig); plt.close(fig)
        with c2:
            st.markdown("**Per-class precision / recall / F1**")
            rep = classification_report(y_enc, y_pred, labels=range(len(class_names)),
                                        target_names=class_names, output_dict=True, zero_division=0)
            rep_df = pd.DataFrame(rep).T
            per_class = rep_df.loc[class_names, ["precision", "recall", "f1-score"]]
            fig, ax = plt.subplots(figsize=(5, 4))
            per_class.plot(kind="bar", ax=ax, ylim=(0, 1.18), rot=0, colormap="Set2")
            ax.set_ylabel("score"); ax.legend(loc="upper center", ncol=3, fontsize=8)
            st.pyplot(fig); plt.close(fig)

        st.markdown("**Classification report**")
        st.dataframe(rep_df.round(3).rename(columns={"f1-score": "f1", "support": "support"}))

        out = pd.DataFrame({"actual": [class_names[i] for i in y_enc],
                            "predicted": [class_names[i] for i in y_pred]})
        st.download_button("Download predictions (CSV)", out.to_csv(index=False).encode(),
                           file_name=f"{chosen.lower().replace(' ', '_')}_predictions.csv")

# ---------------------------------------------------------------- comparison
with tab_compare:
    if X is None:
        st.info("Upload a valid test CSV (or use the bundled one) from the sidebar to compare models.")
    else:
        table, _ = score_all(X, y_enc)
        st.subheader("All five models on the same test file")
        st.dataframe(table.style.format({c: "{:.4f}" for c in METRIC_COLS}, na_rep="N/A")
                     .highlight_max(subset=METRIC_COLS, color="#bfe3c0"),
                     hide_index=True)
        best = table.sort_values(["MCC", "F1"], ascending=False).iloc[0]
        st.success(f"Highest MCC on this file: **{best['ML Model']}** "
                   f"(MCC {best['MCC']:.4f}, F1 {best['F1']:.4f}, Accuracy {best['Accuracy']:.4f}).")
        long = table.melt(id_vars="ML Model", var_name="Metric", value_name="Score")
        fig, ax = plt.subplots(figsize=(11, 4.2))
        sns.barplot(data=long, x="ML Model", y="Score", hue="Metric", ax=ax, palette="viridis")
        ax.set_ylim(0, 1.15); ax.set_xlabel(""); ax.legend(ncol=6, loc="upper center", fontsize=8)
        st.pyplot(fig); plt.close(fig)
        st.download_button("Download comparison table (CSV)", table.to_csv(index=False).encode(),
                           file_name="model_comparison_uploaded.csv")