# =========================================================
# Breast Cancer Wisconsin Classification - FINAL COMPLETE CODE
# (Includes Test CSV Saving + Model Saving for Streamlit)
# =========================================================

# ----------------------------
# 1. IMPORT LIBRARIES
# ----------------------------
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

# ----------------------------
# 2. CREATE MODEL FOLDER
# ----------------------------
os.makedirs("model", exist_ok=True)

# ----------------------------
# 3. LOAD DATASET
# ----------------------------
data = pd.read_csv("data.csv")

# ----------------------------
# 4. CLEAN DATASET
# ----------------------------
data = data.dropna(axis=1, how='all')     # Remove empty columns
data = data.drop(columns=[data.columns[0]])   # Drop ID column

# ----------------------------
# 5. TARGET VARIABLE
# ----------------------------
y = data.iloc[:, 0].map({'M': 1, 'B': 0})
y = y.astype(int)

# ----------------------------
# 6. FEATURES
# ----------------------------
X = data.iloc[:, 1:]
X = X.apply(pd.to_numeric, errors='coerce')

imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# ----------------------------
# DEBUG CHECK
# ----------------------------
print("Target values:", np.unique(y))
print("Target dtype:", y.dtype)

# ----------------------------
# 7. TRAIN TEST SPLIT
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.15,
    random_state=42,
    stratify=y
)

# ----------------------------
# 8. SAVE TEST DATASET
# ----------------------------
test_df = X_test.copy()
test_df["Diagnosis"] = y_test.values
test_df.to_csv("test.csv", index=False)
print("\nTest dataset saved as test.csv")

# ----------------------------
# 9. SCALING
# ----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, "model/scaler.pkl")

# ----------------------------
# 10. DEFINE MODELS
# ----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
}

# ----------------------------
# 11. EVALUATION FUNCTION
# ----------------------------
def evaluate_model(name, model, scaled=False):

    if scaled:
        X_tr, X_te = X_train_scaled, X_test_scaled
    else:
        X_tr, X_te = X_train, X_test

    model.fit(X_tr, y_train)

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    results = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

    return results

# ----------------------------
# 12. TRAIN + SAVE ALL MODELS
# ----------------------------
results = []

for name, model in models.items():

    if name in ["Logistic Regression", "KNN", "XGBoost"]:
        res = evaluate_model(name, model, scaled=True)
        model.fit(X_train_scaled, y_train)
    else:
        res = evaluate_model(name, model, scaled=False)
        model.fit(X_train, y_train)

    # Save model
    filename = name.lower().replace(" ", "_") + ".pkl"
    joblib.dump(model, f"model/{filename}")

    print(f"{name} saved as model/{filename}")

    results.append(res)

# ----------------------------
# 13. SHOW RESULTS
# ----------------------------
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="Accuracy", ascending=False)

print("\n================ MODEL RESULTS ================\n")
print(results_df.to_string(index=False))
print("\n==============================================")
