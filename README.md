# Student Success Radar

A multiclass classification project that compares five models for predicting whether a student will **drop out**, remain **enrolled**, or **graduate**. The project includes separately saved fitted preprocessors and classifiers that the Streamlit app combines into inference pipelines for labeled test CSV files.

## Problem statement

Use student demographic, application, academic, and macroeconomic features to classify the end-of-course outcome. This is an educational analysis, not a basis for making high-stakes decisions about individual students.

## Dataset description

The project uses the UCI Machine Learning Repository's [Predict Students' Dropout and Academic Success dataset](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success). It contains 4,424 instances, 36 input features, and the `Target` label (`Dropout`, `Enrolled`, or `Graduate`). Features include numeric values, binary flags, and integer-coded nominal categories.

The saved split contains 3,539 training rows and 885 held-out test rows. The labeled held-out test data is in the repository-root `test_data.csv`. The original dataset and training split are excluded from Git by `.gitignore`; obtain the source data from UCI if you need to reproduce training.

## GitHub repository link

[SNagarajan2243/student-dropout-ml](https://github.com/SNagarajan2243/student-dropout-ml)

## Models used

Five classifiers are tuned using stratified five-fold cross-validation on the training split, with macro-F1 as the selection score:

- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes
- Support Vector Machine (RBF kernel)

Nominal features are one-hot encoded, binary features are retained, and numeric features are imputed; numeric scaling is applied for models that benefit from it. Preprocessing is fitted within each training pipeline, then the fitted preprocessor and classifier are saved as separate artifacts. This keeps inference consistent with training without fitting anything on uploaded test data.

### Model comparison

The table below reproduces the saved held-out evaluation in `results/model_comparison.csv`. Precision, recall, and F1 are macro-averaged; multiclass AUC is macro one-vs-rest.

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7785 | 0.8879 | 0.7366 | 0.6993 | 0.7112 | 0.6328 |
| Decision Tree | 0.7446 | 0.8490 | 0.6962 | 0.6534 | 0.6641 | 0.5744 |
| KNN | 0.7096 | 0.8025 | 0.6639 | 0.6146 | 0.6263 | 0.5152 |
| Naive Bayes | 0.7040 | 0.8197 | 0.6633 | 0.6365 | 0.6438 | 0.5128 |
| Support Vector Machine | 0.7819 | 0.8887 | 0.7456 | 0.7016 | 0.7154 | 0.6387 |

### Observations and selected model

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Its accuracy (0.7785), AUC (0.8879), macro-F1 (0.7112), and MCC (0.6328) are close to SVM's leading results. It is a strong simpler baseline, although its test scores are slightly lower than SVM's on each of those metrics. |
| Decision Tree | It reaches 0.7446 accuracy and 0.6641 macro-F1, below Logistic Regression and SVM. Its scores are higher than KNN and Naive Bayes on all six reported metrics, making it the middle performer in this comparison. |
| KNN | It has the lowest AUC (0.8025), macro-recall (0.6146), and macro-F1 (0.6263). Its accuracy (0.7096) is higher than Naive Bayes's (0.7040), but lower than the other three models. It therefore does not rank last on every metric. |
| Naive Bayes | It records the lowest accuracy (0.7040) and MCC (0.5128), but its AUC (0.8197), macro-recall (0.6365), and macro-F1 (0.6438) are higher than KNN's. Its relative performance depends on the metric. |
| Support Vector Machine | It leads in accuracy (0.7819), AUC (0.8887), macro-precision (0.7456), macro-recall (0.7016), macro-F1 (0.7154), and MCC (0.6387). Its advantage over Logistic Regression is modest; this test set does not establish a large performance gap. |
| Overall winner for this dataset | **Support Vector Machine (SVM)**, based on its highest MCC and macro-F1, supported by the best accuracy and AUC. MCC and macro-averaged metrics complement accuracy for this imbalanced three-class target. |

These observations describe the saved held-out test evaluation, not a guarantee of performance on new data.

## Repository structure

```text
student-dropout-ml/
├── app.py
├── test_data.csv
├── model/
│   ├── decision_tree.joblib
│   ├── decision_tree_preprocessor.joblib
│   ├── logistic_regression.joblib
│   ├── logistic_regression_preprocessor.joblib
│   ├── metadata.json
│   ├── naive_bayes.joblib
│   ├── naive_bayes_preprocessor.joblib
│   ├── knn.joblib
│   ├── knn_preprocessor.joblib
│   ├── svm.joblib
│   └── svm_preprocessor.joblib
├── notebooks/
│   └── model_experimentation.ipynb
├── results/
│   └── model_comparison.csv
├── src/
│   ├── evaluate_models.py
│   ├── preprocessing.py
│   └── train_models.py
└── requirements.txt
```

`data/students_raw.csv` and `data/train_raw.csv` are intentionally ignored by Git. The root-level `test_data.csv` is included as the bundled labeled evaluation file. Each classifier and its fitted preprocessing transformer are saved separately and paired using `model/metadata.json`. Preprocessing is model-specific: for example, numeric scaling is used for distance- and margin-based models but not for the decision tree. This preserves the fitted training transforms without an unused shared scaler. The saved artifacts were produced with scikit-learn 1.9.1; use the pinned version in `requirements.txt` when loading them.

## Install and run locally

From this directory, create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate` instead. Start the app from the project directory:

```bash
streamlit run app.py
```

In the app, choose **Use bundled test_data.csv** or upload a labeled CSV with all 36 feature columns and the `Target` column. The app validates the columns and target labels, then combines each saved preprocessor and classifier for inference, metrics, confusion matrices, and classification reports. It does not retrain models or fit preprocessing on uploaded data.

## Technologies

Python, Pandas, NumPy, scikit-learn, Joblib, Matplotlib, Seaborn, and Streamlit.

## Links and submission evidence

- **Streamlit application:** [Student Success Radar](https://student-dropout-ml-a6uzhgf8aosgntyzlabqec.streamlit.app/)
