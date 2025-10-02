# prepare_titanic_clean.py
import pandas as pd
import numpy as np

def load_titanic():
    """Try seaborn dataset first; fall back to local CSV."""
    try:
        import seaborn as sns
        return sns.load_dataset("titanic")
    except Exception as e:
        print("[Info] Could not load seaborn titanic:", e)
        # Fallback: adjust the path to where your CSV lives.
        # If you downloaded Kaggle Titanic, you likely have 'train.csv'
        path_options = ["titanic.csv", "train.csv", "./data/titanic.csv", "./data/train.csv"]
        for p in path_options:
            try:
                df = pd.read_csv(p)
                print(f"[Info] Loaded local CSV: {p}")
                # If using Kaggle 'train.csv', rename columns to match common names
                # Kaggle columns: Survived,Pclass,Sex,Age,SibSp,Parch,Fare,Embarked
                rename_map = {
                    "Survived": "survived",
                    "Pclass": "pclass",
                    "Sex": "sex",
                    "Age": "age",
                    "SibSp": "sibsp",
                    "Parch": "parch",
                    "Fare": "fare",
                    "Embarked": "embarked",
                }
                df = df.rename(columns=rename_map)
                return df
            except Exception:
                continue
        raise FileNotFoundError(
            "No Titanic data found. Install seaborn (and be online) or place titanic.csv/train.csv next to this script."
        )

def clean_titanic(df: pd.DataFrame) -> pd.DataFrame:
    """Select useful columns and impute missing values."""
    useful_cols = ["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
    # Keep only columns that exist in the loaded dataset
    existing = [c for c in useful_cols if c in df.columns]
    df = df[existing].copy()

    # Numeric imputations: median
    for col in ["age", "fare"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Categorical imputation for 'embarked': representative sampling (preserve distribution)
    if "embarked" in df.columns:
        embark_dist = df["embarked"].value_counts(normalize=True, dropna=True)
        if not embark_dist.empty:
            cats, probs = embark_dist.index.to_list(), np.array(embark_dist.values)
            mask = df["embarked"].isna()
            if mask.any():
                df.loc[mask, "embarked"] = np.random.choice(cats, size=mask.sum(), p=probs)

    # Type tidy-ups if present
    for c in ["pclass", "sibsp", "parch"]:
        if c in df.columns:
            df[c] = df[c].astype("Int64")  # nullable int if any residual NaNs

    return df

if __name__ == "__main__":
    df_raw = load_titanic()
    print("Raw shape:", df_raw.shape)
    df_clean = clean_titanic(df_raw)
    print("Clean shape:", df_clean.shape)
    print("\nMissing values after cleaning:\n", df_clean.isna().sum())

    out = "titanic_clean.csv"
    df_clean.to_csv(out, index=False)
    print(f"\nSaved cleaned dataset → {out}")
