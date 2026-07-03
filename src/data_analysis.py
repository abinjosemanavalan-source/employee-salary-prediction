import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "employees.csv"

print("Path:", DATA_PATH)
print("Exists:", DATA_PATH.exists())

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("DATASET INFO")
print("=" * 60)
print(df.info())

print("\nFIRST 5 ROWS")
print(df.head())

print("\nMISSING VALUES")
print(df.isnull().sum())

from config import DATASET_PATH