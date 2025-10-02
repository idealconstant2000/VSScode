# day2_baseline.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    RocCurveDisplay,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

# --------------------------
# 0) Load data
# --------------------------
def load_data():
    # Prefer a cleaned CSV if you created it yesterday
    candidates = ["titanic_clean.csv", "./data/titanic_clean.csv"]
    for p in candidates:
        if os.path.exists(p):
            print(f"[Info] Loading cleaned CSV: {p}")
            return pd.read_csv(p)

    # Fallback: load from seaborn and quick-clean the essentials
    print("[Info] 'titanic_clean.csv' not found. Using seaborn fallback + quick cleaning.")
    import seaborn as sns
    df = sns.load_dataset("titanic")

    useful_cols = ["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
    df = df[useful_cols].copy()

    # Impute numeric with median
    for col in ["age", "fare"]:
        df[col] = df[col].fillna(df[col].median())
    # Impute embarked with mode (deterministic)
    df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

    # Tidy types
    for c in ["pclass", "sibsp", "parch"]:
        df[c] = df[c].astype("Int64")

    return df

df = load_data()

# --------------------------
# 1) Train / Test split
# --------------------------
target = "survived"
feature_cols = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]

X = df[feature_cols].copy()
y = df[target].astype(int).values

# Identify column types
num_cols = X.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=np.array(y), random_state=42
)

print(f"[Info] Train size: {X_train.shape}, Test size: {X_test.shape}")
print(f"[Info] Numeric cols: {num_cols}")
print(f"[Info] Categorical cols: {cat_cols}")

# --------------------------
# 2) Preprocess
# --------------------------
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

# --------------------------
# 3) Models
# --------------------------
log_reg = Pipeline(
    steps=[
        ("prep", preprocess),
        ("model", LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)),
    ]
)

rf = Pipeline(
    steps=[
        ("prep", preprocess),
        ("model", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        )),
    ]
)

# --------------------------
# 4) Train
# --------------------------
log_reg.fit(X_train, y_train)
rf.fit(X_train, y_train)

# after rf.fit(...) or log_reg.fit(...)
Path("artifacts").mkdir(exist_ok=True)
joblib.dump(rf, "artifacts/baseline.pkl")   # or log_reg
print("Saved → artifacts/baseline.pkl")
loaded = joblib.load("artifacts/baseline.pkl")
print("Loaded OK. Sample preds:", loaded.predict(X_test[:5]))


# # After fitting a model on train:
# y_val = y_test          # better: make an explicit validation split; demo uses test
# y_score = log_reg.predict_proba(X_test)[:, 1]  # or rf.predict_proba(...)

# # If your model has decision_function instead of predict_proba:
# #y_score = clf.decision_function(X_val)

# --- Add this after training (after .fit calls) ---
from sklearn.metrics import f1_score, roc_auc_score

def auc_and_f1(model, X, y):
    # AUC uses the continuous score; F1 uses hard predictions
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X)[:, 1]
    else:
        y_score = model.decision_function(X)
    y_pred = model.predict(X)
    return roc_auc_score(y, y_score), f1_score(y, y_pred)

results = []
for name, m in [("Logistic Regression", log_reg), ("Random Forest", rf)]:
    auc, f1 = auc_and_f1(m, X_test, y_test)
    results.append({"model": name, "ROC_AUC": auc, "F1": f1})

res_df = pd.DataFrame(results).sort_values("ROC_AUC", ascending=False)

print("\n=== Results (test set) ===")
print(res_df.round(3).to_string(index=False))

res_df.to_csv("results_day2.csv", index=False)
print("\nSaved results_day2.csv")



# --------------------------
# 5) Evaluate
# --------------------------
def evaluate(name, model):
    y_pred = model.predict(X_test)
    # Some models don’t expose predict_proba (e.g., SVC). Ours do:
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        # Fallback if only decision_function exists
        y_score = model.decision_function(X_test)

    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred, digits=3))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    try:
        auc = roc_auc_score(y_test, y_score)
        print(f"ROC AUC: {auc:.3f}")
    except Exception:
        pass
    return y_score

lr_score = evaluate("Logistic Regression", log_reg)
rf_score = evaluate("Random Forest", rf)



# --------------------------
# 6) Plot ROC curves
# --------------------------
plt.figure()
RocCurveDisplay.from_predictions(y_test, lr_score, name="LogReg")
RocCurveDisplay.from_predictions(y_test, rf_score, name="RandomForest")
plt.title("ROC Curves")
plt.grid(True, alpha=0.3)
plt.show()
