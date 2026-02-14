import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# -----------------------------
# PAGE TITLE
# -----------------------------
st.title("Breast Cancer Classification App")

# -----------------------------
# LOAD MODELS + SCALER
# -----------------------------
@st.cache_resource
def load_models():

    models = {
        "Logistic Regression": joblib.load("model/logistic_regression.pkl"),
        "Decision Tree": joblib.load("model/decision_tree.pkl"),
        "KNN": joblib.load("model/knn.pkl"),
        "Naive Bayes": joblib.load("model/naive_bayes.pkl"),
        "Random Forest": joblib.load("model/random_forest.pkl"),
        "XGBoost": joblib.load("model/xgboost.pkl")
    }

    scaler = joblib.load("model/scaler.pkl")

    return models, scaler

models, scaler = load_models()

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Test CSV File",
    type=["csv"]
)

# -----------------------------
# MODEL SELECT
# -----------------------------
model_name = st.selectbox(
    "Select Model",
    list(models.keys())
)

# -----------------------------
# PREDICT BUTTON
# -----------------------------
if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    if "Diagnosis" not in data.columns:
        st.error("CSV must contain Diagnosis column")
    else:

        X = data.drop("Diagnosis", axis=1)
        y_true = data["Diagnosis"]

        # Scale if needed
        if model_name in ["Logistic Regression", "KNN", "XGBoost"]:
            X = scaler.transform(X)

        model = models[model_name]

        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]

        # -----------------------------
        # METRICS
        # -----------------------------
        acc = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_prob)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)

        st.subheader("Evaluation Metrics")

        st.write(f"Accuracy: {acc:.4f}")
        st.write(f"AUC: {auc:.4f}")
        st.write(f"Precision: {prec:.4f}")
        st.write(f"Recall: {rec:.4f}")
        st.write(f"F1 Score: {f1:.4f}")
        st.write(f"MCC: {mcc:.4f}")

        # -----------------------------
        # CONFUSION MATRIX
        # -----------------------------
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        st.write(cm)

        # -----------------------------
        # CLASSIFICATION REPORT
        # -----------------------------
        st.subheader("Classification Report")
        st.text(classification_report(y_true, y_pred))
