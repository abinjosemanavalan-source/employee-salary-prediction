import joblib
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import r2_score

from src.feature_engineering import FeatureEngineering
from src.data_splitter import DataSplitter


class BestModelTrainer:

    def __init__(self, X_train, X_test, y_train, y_test):

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Random Forest": RandomForestRegressor(
                random_state=42,
                n_estimators=100
            ),
            "Gradient Boosting": GradientBoostingRegressor(
                random_state=42
            )
        }

    def train_and_save(self):

        best_model = None
        best_score = float("-inf")
        best_name = ""

        for name, model in self.models.items():

            model.fit(self.X_train, self.y_train)

            predictions = model.predict(self.X_test)

            score = r2_score(self.y_test, predictions)

            print(f"{name:20} -> R² = {score:.4f}")

            if score > best_score:

                best_score = score
                best_model = model
                best_name = name

        Path("models").mkdir(exist_ok=True)

        joblib.dump(best_model, "models/best_model.pkl")

        print("\n==============================")
        print("🏆 Best Model Saved")
        print("==============================")
        print(f"Model : {best_name}")
        print(f"R²    : {best_score:.4f}")


if __name__ == "__main__":

    DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "employee_salary.csv"

    df = pd.read_csv(DATA_PATH)

    feature = FeatureEngineering(df)

    feature.drop_columns()
    feature.encode_features()

    X, y = feature.split_features_target()

    splitter = DataSplitter(X, y)

    X_train, X_test, y_train, y_test = splitter.split()

    trainer = BestModelTrainer(
        X_train,
        X_test,
        y_train,
        y_test
    )

    trainer.train_and_save()