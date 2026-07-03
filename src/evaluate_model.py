import pandas as pd
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from feature_engineering import FeatureEngineering
from data_splitter import DataSplitter


class ModelEvaluator:

    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

    def evaluate(self):

        predictions = self.model.predict(self.X_test)

        mae = mean_absolute_error(self.y_test, predictions)
        mse = mean_squared_error(self.y_test, predictions)
        rmse = mse ** 0.5
        r2 = r2_score(self.y_test, predictions)

        print("\n========== MODEL EVALUATION ==========\n")

        print(f"MAE  : {mae:.2f}")
        print(f"MSE  : {mse:.2f}")
        print(f"RMSE : {rmse:.2f}")
        print(f"R²   : {r2:.4f}")

        return predictions


if __name__ == "__main__":

    DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "employee_salary.csv"

    df = pd.read_csv(DATA_PATH)

    feature = FeatureEngineering(df)

    feature.drop_columns()
    feature.encode_features()

    X, y = feature.split_features_target()

    splitter = DataSplitter(X, y)

    X_train, X_test, y_train, y_test = splitter.split()

    model = LinearRegression()

    model.fit(X_train, y_train)

    evaluator = ModelEvaluator(model, X_test, y_test)

    evaluator.evaluate()