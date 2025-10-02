# class_wk1_day1.py
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Load Titanic dataset from Seaborn
df = sns.load_dataset("titanic")

# Basic checks
print("Shape of dataset:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nInfo:")
print(df.info())
print("\nMissing values:\n", df.isnull().sum())

# Simple visualization: survival rate by class
sns.barplot(data=df, x="class", y="survived")
plt.title("Survival Rate by Passenger Class")
plt.show()

# Select useful columns
# useful_cols = ["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
# df = df[useful_cols]

# print("Missing values before handling:")
# print(df.isnull().sum(), "\n")

# # Handle missing values
# df["age"].fillna(df["age"].median(), inplace=True)
# df["fare"].fillna(df["fare"].median(), inplace=True)
# df["embarked"].fillna(df["embarked"].mode()[0], inplace=True)

# print("Missing values after handling:")
# print(df.isnull().sum(), "\n")

# # Quick check
# print("First 5 cleaned rows:")
# print(df.head())
