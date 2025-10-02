# day4_tuning_and_reporting.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import os, json, platform, warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RepeatedStratifiedKFold,
    RandomizedSearchCV,
    cross_val_score,
)
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    ConfusionMatrixDisplay,
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib
import sklearn

warnings.filterwarnings("ignore", category=UserWarning)

# =========================
# CONFIG
# =========================
RANDOM_STATE = 42
TEST_SIZE = 0.20                 # hold-out test fraction
VAL_FRACTION_OF_TRAIN = 0.20     # validation fraction inside the train split
MAIN_SCORING = "roc_auc"         # or "average_precision" for PR-AUC
N_SPLITS = 5
N_REPEATS = 2                    # for more stable CV estimates
N_ITER_SEARCH = 25               # RandomizedSearch iterations
ART_DIR = Path("artifacts")
RESULTS_CSV = Path("results_day4.csv")

np.random.seed(RANDOM_STATE)

# =========================
# DATA LOADING
# =========================
def load_data() -> pd.DataFrame:
    for p in ["titanic_clean.csv", "./data/titanic_clean.csv"]:
        if os.path.exists(p):
            print(f"[Info] Loading cleaned CSV: {p}")
            return pd.read_csv(p)

    print("[Info] titanic_clean.csv not found. Using seaborn fallback + quick cleaning.")
    try:
        import seaborn as sns
        df = sns.load_dataset("titanic")
    except Exception as e:
        raise SystemExit(
            "No titanic_clean.csv and seaborn fallback failed. "
            "Place titanic_clean.csv next to this script or install seaborn."
        ) from e

    cols = ["survived","pclass","sex","age","sibsp","parch","fare","embarked"]
    df = df[cols].copy()

    for col in ["age","fare"]:
        df[col] = df[col].fillna(df[col].median())
    df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

    for c in ["pclass","sibsp","parch"]:
        df[c] = df[c].astype("Int64")
    return df

# =========================
# PREPROCESSOR
# =========================
def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    return pre, num_cols, cat_cols

# =========================
# CV HELPERS
# =========================
def crossval_scores(pipe: Pipeline, X, y, scoring="roc_auc") -> tuple[float,float]:
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return float(np.mean(scores)), float(np.std(scores))

def crossval_scores_repeated(pipe: Pipeline, X, y, scoring="roc_auc") -> tuple[float,float]:
    cv_r = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE
    )
    scores = cross_val_score(pipe, X, y, cv=cv_r, scoring=scoring, n_jobs=-1)
    return float(np.mean(scores)), float(np.std(scores))

# =========================
# THRESHOLDING
# =========================
def select_threshold_f1(y_true, y_score) -> float:
    """Pick tau that maximizes F1 on validation set."""
    p, r, t = precision_recall_curve(y_true, y_score)
    p, r, t = p[:-1], r[:-1], t
    f1 = 2 * p * r / (p + r + 1e-12)
    return float(t[np.argmax(f1)])

def select_threshold_with_constraints(y_true, y_score, min_precision=None, min_recall=None) -> float | None:
    """Optionally, choose tau to satisfy precision or recall constraints on validation."""
    p, r, t = precision_recall_curve(y_true, y_score)
    p, r, t = p[:-1], r[:-1], t
    if min_precision is not None:
        mask = p >= min_precision
        if np.any(mask):
            idx = np.argmax(r[mask])  # highest recall under precision constraint
            return float(t[mask][idx])
    if min_recall is not None:
        mask = r >= min_recall
        if np.any(mask):
            idx = np.argmax(p[mask])  # highest precision under recall constraint
            return float(t[mask][idx])
    return None

# =========================
# BOOTSTRAP CI
# =========================
def bootstrap_ci_auc_f1(y_true, y_score, y_pred, B=1000, alpha=0.05, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs, f1s = [], []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        try:
            aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
        except ValueError:
            continue
        f1s.append(f1_score(y_true[idx], y_pred[idx]))
    def ci(arr):
        arr = np.array(arr)
        lo, hi = np.percentile(arr, [100*alpha/2, 100*(1-alpha/2)])
        return float(arr.mean()), float(lo), float(hi)
    return {"AUC": ci(aucs), "F1": ci(f1s)}

# =========================
# PLOTTING
# =========================
def save_roc_pr_curves(y_true, y_score, name_prefix: str):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC — {name_prefix}")
    ART_DIR.mkdir(parents=True, exist_ok=True)
    roc_path = ART_DIR / f"{name_prefix}_roc.png"
    plt.savefig(roc_path, dpi=150, bbox_inches="tight"); plt.close()

    prec, rec, _ = precision_recall_curve(y_true, y_score)
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR — {name_prefix}")
    pr_path = ART_DIR / f"{name_prefix}_pr.png"
    plt.savefig(pr_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[Info] Saved curves → {roc_path.name}, {pr_path.name}")

def save_confusion(y_true, y_pred, name_prefix: str):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(values_format="d")
    plt.title(f"Confusion Matrix — {name_prefix}")
    path = ART_DIR / f"{name_prefix}_cm.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[Info] Saved confusion matrix → {path.name}")

# =========================
# IMPORTANCES
# =========================
def model_based_importance(pipe: Pipeline, name_prefix: str, num_cols, cat_cols):
    model = pipe.named_steps["model"]
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        print(f"[Info] {name_prefix}: model has no feature_importances_. Skipping.")
        return
    pre = pipe.named_steps["prep"]
    try:
        feat_names = pre.get_feature_names_out()
        feat_names = [n.replace("num__","").replace("cat__","") for n in feat_names]
    except Exception:
        ohe = pre.named_transformers_["cat"]
        feat_names = list(num_cols) + list(ohe.get_feature_names_out(cat_cols))
    imp = pd.Series(importances, index=feat_names).sort_values(ascending=False)
    path = ART_DIR / f"{name_prefix}_model_importance.csv"
    imp.to_csv(path, header=["importance"])
    print(f"[Info] Saved model-based importances → {path.name}")

def permutation_importance_raw(pipe: Pipeline, X_test: pd.DataFrame, y_test: np.ndarray, name_prefix: str):
    r = permutation_importance(pipe, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, scoring=MAIN_SCORING, n_jobs=-1)
    imp = pd.Series(r.importances_mean, index=X_test.columns).sort_values(ascending=False)
    path = ART_DIR / f"{name_prefix}_perm_importance_raw.csv"
    imp.to_csv(path, header=["importance"])
    print(f"[Info] Saved permutation importance (raw) → {path.name}")

# =========================
# LOGGING
# =========================
def log_results(row: dict, path: Path = RESULTS_CSV):
    exists = path.exists()
    df = pd.DataFrame([row])
    df.to_csv(path, mode="a" if exists else "w", header=not exists, index=False)
    print(f"[Info] Appended results → {path}")

# =========================
# MAIN
# =========================
def main():
    df = load_data()
    target = "survived"
    features = ["pclass","sex","age","sibsp","parch","fare","embarked"]

    X = df[features].copy()
    y = df[target].astype(int).values

    # Train/Test split
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    # Validation split inside train
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=VAL_FRACTION_OF_TRAIN,
        stratify=y_trainval, random_state=RANDOM_STATE
    )

    print(f"[Info] Splits — train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")

    pre, num_cols, cat_cols = build_preprocessor(X)

    # Models & search spaces
    rf_pipe = Pipeline([("prep", pre), ("model", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))])
    gb_pipe = Pipeline([("prep", pre), ("model", GradientBoostingClassifier(random_state=RANDOM_STATE))])

    rf_space = {
        "model__n_estimators": np.arange(200, 801, 100),
        "model__max_depth": [None, 4, 6, 8, 10, 12],
        "model__min_samples_leaf": [1, 2, 4, 6, 8],
        "model__max_features": ["sqrt", "log2", 0.5, 1.0],
    }
    gb_space = {
        "model__n_estimators": np.arange(100, 501, 50),
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "model__max_depth": [2, 3, 4],
        "model__min_samples_leaf": [1, 2, 4],
    }

    # Robust CV estimates (repeated)
    for name, pipe in [("RF (base)", rf_pipe), ("GB (base)", gb_pipe)]:
        mean_cv, std_cv = crossval_scores_repeated(pipe, X_train, y_train, scoring=MAIN_SCORING)
        print(f"[CV] {name}: {MAIN_SCORING} mean={mean_cv:.3f} ± {std_cv:.3f}")

    # RandomizedSearchCV tuning on train
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rf_search = RandomizedSearchCV(
        rf_pipe, rf_space, n_iter=N_ITER_SEARCH, cv=cv, scoring=MAIN_SCORING,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=0
    ).fit(X_train, y_train)
    gb_search = RandomizedSearchCV(
        gb_pipe, gb_space, n_iter=N_ITER_SEARCH, cv=cv, scoring=MAIN_SCORING,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=0
    ).fit(X_train, y_train)

    # Choose champion by CV score
    rf_best, gb_best = rf_search.best_estimator_, gb_search.best_estimator_
    rf_cv, gb_cv = rf_search.best_score_, gb_search.best_score_
    champ_name, champ = ("RF_Tuned", rf_best) if rf_cv >= gb_cv else ("GB_Tuned", gb_best)
    print(f"[Tune] Champion by CV: {champ_name} (CV {MAIN_SCORING}={max(rf_cv, gb_cv):.3f})")

    # Fit champion on TRAIN ONLY (not including val) to compute threshold on val
    champ.fit(X_train, y_train)

    # Select threshold on validation set (maximize F1 by default)
    y_val_score = champ.predict_proba(X_val)[:, 1] if hasattr(champ, "predict_proba") else champ.decision_function(X_val)
    tau = select_threshold_f1(y_val, y_val_score)
    print(f"[Thresh] Selected τ (max F1 on val) = {tau:.3f}")

    # Refit champion on TRAIN+VAL, then final TEST evaluation using locked τ
    champ.fit(X_trainval, y_trainval)
    y_test_score = champ.predict_proba(X_test)[:, 1] if hasattr(champ, "predict_proba") else champ.decision_function(X_test)
    y_test_pred = (y_test_score >= tau).astype(int)

    # Metrics on test
    auc = roc_auc_score(y_test, y_test_score)
    f1  = f1_score(y_test, y_test_pred)
    prec = precision_score(y_test, y_test_pred)
    rec  = recall_score(y_test, y_test_pred)
    ap   = average_precision_score(y_test, y_test_score)  # PR-AUC

    print("\n=== Final Test Metrics (champion with locked τ) ===")
    print(f"AUC={auc:.3f} | PR-AUC={ap:.3f} | F1={f1:.3f} | Precision={prec:.3f} | Recall={rec:.3f}")
    print(classification_report(y_test, y_test_pred, digits=3))

    # Bootstrap CIs
    ci = bootstrap_ci_auc_f1(y_test, y_test_score, y_test_pred, B=1000, alpha=0.05, seed=RANDOM_STATE)
    print(f"[CI] AUC mean/95% CI: {ci['AUC']}")
    print(f"[CI] F1  mean/95% CI: {ci['F1']}")

    # Artifacts
    ART_DIR.mkdir(parents=True, exist_ok=True)
    save_roc_pr_curves(y_test, y_test_score, champ_name)
    save_confusion(y_test, y_test_pred, champ_name)
    # Importances
    pre, num_cols, cat_cols = build_preprocessor(X)  # for names
    model_based_importance(champ, champ_name, num_cols, cat_cols)
    permutation_importance_raw(champ, X_test, y_test, champ_name)

    # Save model + threshold
    model_path = ART_DIR / "day4_best.pkl"
    joblib.dump(champ, model_path, compress=3)
    with open(ART_DIR / "threshold.json", "w") as f:
        json.dump({"tau": float(tau)}, f)
    print(f"[Save] Model → {model_path.name}, Threshold → threshold.json")

    # Log a row to CSV
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "model": champ_name,
        "cv_score": max(rf_cv, gb_cv),
        "test_auc": auc,
        "test_pr_auc": ap,
        "test_f1": f1,
        "test_precision": prec,
        "test_recall": rec,
        "threshold": float(tau),
        "test_size": TEST_SIZE,
        "n_splits": N_SPLITS,
        "n_repeats": N_REPEATS,
        "scoring": MAIN_SCORING,
        "seed": RANDOM_STATE,
    }
    log_results(row, RESULTS_CSV)

if __name__ == "__main__":
    main()
