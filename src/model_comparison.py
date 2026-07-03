import pandas as pd
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.feature_engineering import FeatureEngineering
from data_splitter import DataSplitter


class ModelComparison:

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

    def compare_models(self):

        results = []

        print("\n========== MODEL COMPARISON ==========\n")

        for name, model in self.models.items():

            model.fit(self.X_train, self.y_train)

            predictions = model.predict(self.X_test)

            mae = mean_absolute_error(self.y_test, predictions)
            mse = mean_squared_error(self.y_test, predictions)
            rmse = mse ** 0.5
            r2 = r2_score(self.y_test, predictions)

            results.append({
                "Model": name,
                "MAE": round(mae, 2),
                "RMSE": round(rmse, 2),
                "R2 Score": round(r2, 4)
            })

        results_df = pd.DataFrame(results)

        results_df = results_df.sort_values(
            by="R2 Score",
            ascending=False
        )

        print(results_df)

        best_model = results_df.iloc[0]["Model"]

        print("\n🏆 Best Model:", best_model)

        return results_df


if __name__ == "__main__":

    DATA_PATH = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "employee_salary.csv"
    )

    df = pd.read_csv(DATA_PATH)

    feature = FeatureEngineering(df)

    feature.drop_columns()
    feature.encode_features()

    X, y = feature.split_features_target()

    splitter = DataSplitter(X, y)

    X_train, X_test, y_train, y_test = splitter.split()

    comparison = ModelComparison(
        X_train,
        X_test,
        y_train,
        y_test
    )

    comparison.compare_models()