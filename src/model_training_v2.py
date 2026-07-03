import pandas as pd
from pathlib import Path
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =====================================
# Load Dataset
# =====================================

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "employees.csv"

df = pd.read_csv(DATA_PATH)

# =====================================
# Drop unwanted columns
# =====================================

df.drop(
    columns=[
        "Start Date",
        "Last Login Time"
    ],
    inplace=True
)

# =====================================
# Encode categorical columns
# =====================================

encoder = LabelEncoder()

for col in [
    "Gender",
    "Senior Management",
    "Team"
]:
    df[col] = encoder.fit_transform(df[col])

# =====================================
# Features & Target
# =====================================

X = df.drop("Salary", axis=1)

y = df["Salary"]

# =====================================
# Train Test Split
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("="*50)
print("Training Samples :", len(X_train))
print("Testing Samples  :", len(X_test))
print("="*50)

# =====================================
# Models
# =====================================

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

best_model = None
best_score = -999

print("\nMODEL PERFORMANCE\n")

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    print(f"{name:20} R² = {r2:.4f}")

    if r2 > best_score:
        best_score = r2
        best_model = model
        best_name = name

# =====================================
# Save Best Model
# =====================================

joblib.dump(
    best_model,
    "best_model_v2.pkl"
)

print("\n"+"="*50)

print("🏆 BEST MODEL")

print("="*50)

print("Model :", best_name)

print("R²    :", round(best_score,4))

print("\nModel saved as best_model_v2.pkl")