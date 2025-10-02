# day3_trees_and_tuning.py
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.inspection import permutation_importance

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# --------------------------
# 0) Load data (prefer cleaned CSV)
# --------------------------
def load_data():
    for p in ["titanic_clean.csv", "./data/titanic_clean.csv"]:
        if os.path.exists(p):
            print(f"[Info] Loading cleaned CSV: {p}")
            return pd.read_csv(p)

    # Fallback: minimal cleaner (requires seaborn)
    print("[Info] titanic_clean.csv not found. Using seaborn fallback + quick cleaning.")
    try:
        import seaborn as sns
        df = sns.load_dataset("titanic")
    except Exception as e:
        raise SystemExit("No titanic_clean.csv and seaborn fallback failed. "
                         "Place titanic_clean.csv next to this script or install seaborn.") from e

    cols = ["survived","pclass","sex","age","sibsp","parch","fare","embarked"]
    df = df[cols].copy()
    for col in ["age","fare"]:
        df[col] = df[col].fillna(df[col].median())
    df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
    for c in ["pclass","sibsp","parch"]:
        df[c] = df[c].astype("Int64")
    return df

df = load_data()
target = "survived"
features = ["pclass","sex","age","sibsp","parch","fare","embarked"]

X = df[features].copy()
y = df[target].astype(int).values

num_cols = X.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=np.array(y), random_state=RANDOM_STATE
)
print(f"[Info] Train: {X_train.shape}, Test: {X_test.shape}")
print(f"[Info] Numeric: {num_cols} | Categorical: {cat_cols}")

# --------------------------
# 1) Preprocess
# --------------------------
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

# --------------------------
# 2) Models
# --------------------------
models = {
    "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE)
}

pipelines = {
    name: Pipeline([("prep", preprocess), ("model", model)])
    for name, model in models.items()
}

# --------------------------
# 3) Cross-validated comparison
# --------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

rows = []
for name, pipe in pipelines.items():
    auc_cv = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    f1_cv  = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
    rows.append({
        "model": name,
        "cv_auc_mean": auc_cv.mean(),
        "cv_auc_std":  auc_cv.std(),
        "cv_f1_mean":  f1_cv.mean(),
        "cv_f1_std":   f1_cv.std(),
    })
cv_df = pd.DataFrame(rows).sort_values("cv_auc_mean", ascending=False)
print("\n=== CV Results (train folds) ===")
print(cv_df.round(3).to_string(index=False))

# --------------------------
# 4) Fit on train, evaluate on test
# --------------------------
test_rows = []
fitted = {}
for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)
    fitted[name] = pipe

    y_pred = pipe.predict(X_test)
    try:
        y_score = pipe.predict_proba(X_test)[:,1]
    except AttributeError:
        # some models may not have predict_proba
        try:
            y_score = pipe.decision_function(X_test)
        except AttributeError:
            y_score = y_pred  # fallback (AUC will be less meaningful)

    auc = roc_auc_score(y_test, y_score)
    f1  = f1_score(y_test, y_pred)

    print(f"\n=== {name} (test) ===")
    print(classification_report(y_test, y_pred, digits=3))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print(f"ROC AUC: {auc:.3f} | F1: {f1:.3f}")

    test_rows.append({"model": name, "test_auc": auc, "test_f1": f1})

test_df = pd.DataFrame(test_rows).sort_values("test_auc", ascending=False)
print("\n=== Test Results ===")
print(test_df.round(3).to_string(index=False))

# Save combined results
res = cv_df.merge(test_df, on="model", how="left")
Path("artifacts").mkdir(exist_ok=True, parents=True)
res.to_csv("results_day3.csv", index=False)
print("\nSaved results_day3.csv")

# --------------------------
# 5) Feature Importance (model-based)
#    (works best for tree models)
# --------------------------
def get_feature_names(fitted_pipeline):
    # Try to get full expanded names
    try:
        names = fitted_pipeline.named_steps["prep"].get_feature_names_out()
        return [n.replace("num__","").replace("cat__","") for n in names]
    except Exception:
        # Fallback: numeric + expanded cat names
        ohe = fitted_pipeline.named_steps["prep"].named_transformers_["cat"]
        cat_names = list(ohe.get_feature_names_out(cat_cols))
        return list(num_cols) + cat_names

def plot_model_importance(name, fitted_pipe, topk=15):
    try:
        model = fitted_pipe.named_steps["model"]
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            print(f"[Info] {name}: no model-based importances (skipping).")
            return
        feat_names = get_feature_names(fitted_pipe)
        imp = pd.Series(importances, index=feat_names).sort_values(ascending=False).head(topk)
        ax = imp.iloc[::-1].plot(kind="barh", figsize=(7,5))
        ax.set_title(f"{name} — Top {topk} Feature Importances")
        plt.tight_layout()
        out = f"artifacts/{name.lower()}_importances.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"[Info] Saved model-based importance plot → {out}")
    except Exception as e:
        print(f"[Warn] Could not compute model-based importances for {name}: {e}")

for name in ["RandomForest","GradientBoosting","DecisionTree"]:
    if name in fitted:
        plot_model_importance(name, fitted[name], topk=15)

# --------------------------
# 6) Permutation Importance (threshold-agnostic, more reliable)
# --------------------------
# --- replace your permutation importance helper with this ---
def permutation_imp_raw(name, fitted_pipe, X_test, y_test, scoring="roc_auc", topk=15):
    r = permutation_importance(
        fitted_pipe,
        X_test, y_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring=scoring,
        n_jobs=-1,
    )
    feat_names = list(X_test.columns)  # raw input feature names (7)
    imp = pd.Series(r["importances_mean"], index=feat_names).sort_values(ascending=False)
    out_csv = f"artifacts/{name.lower()}_perm_importance.csv"
    imp.to_csv(out_csv, header=["importance"])
    print(f"[Info] Saved permutation importance (raw features) → {out_csv}")

# --- calls (replace previous ones) ---
for name in ["RandomForest", "GradientBoosting"]:
    if name in fitted:
        permutation_imp_raw(name, fitted[name], X_test, y_test, scoring="roc_auc", topk=20)


# --------------------------
# 7) RandomizedSearchCV (quick tuning)
# --------------------------
rf_space = {
    "model__n_estimators": np.arange(200, 801, 100),
    "model__max_depth": [None, 4, 6, 8, 10, 12],
    "model__min_samples_leaf": [1, 2, 4, 6, 8],
    "model__max_features": ["sqrt", "log2", 0.5, 0.7, 1.0],
}
gb_space = {
    "model__n_estimators": np.arange(100, 501, 50),
    "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "model__max_depth": [2, 3, 4],
    "model__min_samples_leaf": [1, 2, 4],
}

def tune(name: str, base_model, param_space, n_iter=25):
    pipe = Pipeline([("prep", preprocess), ("model", base_model)])
    search = RandomizedSearchCV(
        pipe, param_distributions=param_space, n_iter=n_iter, cv=cv,
        scoring="roc_auc", random_state=RANDOM_STATE, n_jobs=-1, verbose=0
    )
    search.fit(X_train, y_train)
    print(f"\n=== Tuning {name} ===")
    print("Best AUC (CV):", round(search.best_score_, 4))
    print("Best params:", search.best_params_)
    # test-set eval
    best = search.best_estimator_
    y_pred = best.predict(X_test)  # type: ignore
    if hasattr(best, "predict_proba") and callable(getattr(best, "predict_proba", None)):
        y_score = getattr(best, "predict_proba")(X_test)[:, 1]
    elif hasattr(best, "decision_function") and callable(getattr(best, "decision_function", None)):
        y_score = getattr(best, "decision_function")(X_test)
    else:
        y_score = y_pred  # fallback if neither method exists
    auc = roc_auc_score(y_test, y_score)
    f1  = f1_score(y_test, y_pred)
    print(f"{name} tuned (test): AUC={auc:.3f}, F1={f1:.3f}")
    return best, auc, f1

best_rf, rf_auc, rf_f1 = tune("RandomForest", RandomForestClassifier(random_state=RANDOM_STATE), rf_space, n_iter=25)
best_gb, gb_auc, gb_f1 = tune("GradientBoosting", GradientBoostingClassifier(random_state=RANDOM_STATE), gb_space, n_iter=25)

# --------------------------
# 8) Save tuned winner + append results
# --------------------------
winner_name, winner, w_auc, w_f1 = (
    ("RF_Tuned", best_rf, rf_auc, rf_f1) if rf_auc >= gb_auc else ("GB_Tuned", best_gb, gb_auc, gb_f1)
)

joblib.dump(winner, "artifacts/day3_best.pkl", compress=3)
print(f"\nSaved tuned winner → artifacts/day3_best.pkl ({winner_name}, AUC={w_auc:.3f}, F1={w_f1:.3f})")

more = pd.DataFrame([
    {"model": "RF_Tuned", "test_auc": rf_auc, "test_f1": rf_f1},
    {"model": "GB_Tuned", "test_auc": gb_auc, "test_f1": gb_f1},
])
final = pd.concat([test_df, more], ignore_index=True).sort_values("test_auc", ascending=False)
final.to_csv("results_day3.csv", index=False)  # overwrite with tuned rows added
print("\nUpdated results_day3.csv with tuned models:")
print(final.round(3).to_string(index=False))
