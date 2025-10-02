# Example skeleton you can run/debug in VS Code
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(Xtr, ytr)

print(classification_report(yte, model.predict(Xte)))
joblib.dump(model, "model.pkl")
print("Saved model.pkl")
