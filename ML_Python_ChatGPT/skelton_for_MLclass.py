import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("data.csv")
X = df.drop("target", axis=1); y = df["target"]

num = X.select_dtypes(include="number").columns
cat = X.select_dtypes(exclude="number").columns

pre = ColumnTransformer([
    ("num", StandardScaler(), num),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat)
])

clf = Pipeline([
    ("prep", pre),
    ("model", RandomForestClassifier(n_estimators=300, random_state=42))
])

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
clf.fit(Xtr, ytr)
pred = clf.predict(Xte); proba = clf.predict_proba(Xte)[:,1]
print(classification_report(yte, pred))
print("ROC AUC:", roc_auc_score(yte, proba))

joblib.dump(clf, "model.pkl")
