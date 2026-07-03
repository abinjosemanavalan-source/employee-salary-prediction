import pandas as pd
from pathlib import Path
import joblib

from sklearn.linear_model import LinearRegression

from src.feature_engineering import FeatureEngineering
from src.data_splitter import DataSplitter


class TrainModel:

    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.model = LinearRegression()

    def train(self):

        self.model.fit(self.X_train, self.y_train)

        print("✅ Model Trained Successfully")

        return self.model

    def save_model(self):

        Path("models").mkdir(exist_ok=True)

        joblib.dump(self.model, "models/linear_regression.pkl")

        print("✅ Model Saved Successfully")


if __name__ == "__main__":

    DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "employee_salary.csv"

    df = pd.read_csv(DATA_PATH)

    feature = FeatureEngineering(df)

    feature.drop_columns()

    feature.encode_features()

    X, y = feature.split_features_target()

    splitter = DataSplitter(X, y)

    X_train, X_test, y_train, y_test = splitter.split()

    trainer = TrainModel(X_train, y_train)

    trainer.train()

    trainer.save_model()