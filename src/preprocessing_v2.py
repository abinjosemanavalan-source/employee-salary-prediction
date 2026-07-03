import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Load Dataset
# -----------------------------

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "employees.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("ORIGINAL DATASET")
print("=" * 60)
print(df.head())

# -----------------------------
# Drop unnecessary columns
# -----------------------------

columns_to_drop = [
    "Start Date",
    "Last Login Time"
]

df.drop(columns=columns_to_drop, inplace=True)

print("\n✅ Unnecessary columns removed.")

# -----------------------------
# Encode categorical columns
# -----------------------------

label_encoder = LabelEncoder()

categorical_columns = [
    "Gender",
    "Senior Management",
    "Team"
]

for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])

print("✅ Categorical columns encoded.")

# -----------------------------
# Split Features and Target
# -----------------------------

X = df.drop("Salary", axis=1)

y = df["Salary"]

print("\n" + "=" * 60)
print("FEATURES")
print("=" * 60)
print(X.head())

print("\n" + "=" * 60)
print("TARGET")
print("=" * 60)
print(y.head())