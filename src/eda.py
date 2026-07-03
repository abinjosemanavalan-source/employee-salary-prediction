import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "employees.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 50)
print("DATASET SHAPE")
print("=" * 50)

print(df.shape)

print("\n")

print("=" * 50)
print("SALARY STATISTICS")
print("=" * 50)

print(df["Salary"].describe())

# Salary Distribution
plt.figure(figsize=(8,5))
plt.hist(df["Salary"], bins=20)
plt.title("Salary Distribution")
plt.xlabel("Salary")
plt.ylabel("Number of Employees")
plt.grid(True)

plt.show()

# Team vs Salary
team_salary = df.groupby("Team")["Salary"].mean().sort_values()

plt.figure(figsize=(10,5))
team_salary.plot(kind="bar")
plt.title("Average Salary by Team")
plt.ylabel("Salary")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()