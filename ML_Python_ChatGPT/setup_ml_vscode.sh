#!/bin/zsh
# Setup ML environment for VS Code on macOS

set -e

# 1) Create project folder (use current dir or pass a name: ./setup_ml_vscode.sh my-ml-project)
PROJECT_DIR="${1:-$PWD}"
if [[ "$PROJECT_DIR" != "$PWD" ]]; then
  mkdir -p "$PROJECT_DIR"
  cd "$PROJECT_DIR"
fi

echo "Project: $PROJECT_DIR"

# 2) Create and activate venv
python3 -m venv .venv
source .venv/bin/activate

# 3) Upgrade pip and install packages
python -m pip install --upgrade pip
pip install -U numpy pandas matplotlib scikit-learn fastapi uvicorn joblib

# 4) (Optional) record exact deps
pip freeze > requirements.txt

# 5) VS Code settings so it auto-picks the venv
mkdir -p .vscode
cat > .vscode/settings.json <<'JSON'
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.analysis.typeCheckingMode": "basic",
  "python.testing.pytestEnabled": true,
  "terminal.integrated.defaultProfile.osx": "zsh"
}
JSON

# 6) Launch configuration (debug a simple script or FastAPI app)
cat > .vscode/launch.json <<'JSON'
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Current File",
      "type": "python",
      "request": "launch",
      "program": "${file}"
    },
    {
      "name": "FastAPI (uvicorn main:app)",
      "type": "python",
      "request": "launch",
      "module": "uvicorn",
      "args": ["main:app", "--reload"],
      "jinja": true
    }
  ]
}
JSON

# 7) Recommend useful extensions
cat > .vscode/extensions.json <<'JSON'
{
  "recommendations": [
    "ms-python.python",
    "ms-python.vscode-pylance",
    "ms-toolsai.jupyter",      // optional, for notebooks inside VS Code
    "ms-python.black-formatter",
    "ms-python.isort"
  ]
}
JSON

# 8) Starter files
cat > main.py <<'PY'
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def hello():
    return {"message": "Hello from FastAPI + scikit-learn environment!"}
PY

cat > train_example.py <<'PY'
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
PY

# 9) Open VS Code (requires 'code' command on PATH)
if command -v code >/dev/null 2>&1; then
  code .
else
  echo "Tip: Install VS Code and enable the 'code' command (VS Code > Command Palette > 'Shell Command: Install 'code' command in PATH')."
fi

echo "Done! Activate later with: source .venv/bin/activate"
