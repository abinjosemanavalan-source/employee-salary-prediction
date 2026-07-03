import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from feature_engineering import FeatureEngineering


class DataSplitter:

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def split(self):

        X_train, X_test, y_train, y_test = train_test_split(
            self.X,
            self.y,
            test_size=0.20,
            random_state=42
        )

        print("\n✅ Data Split Completed")

        print(f"Training Samples : {len(X_train)}")
        print(f"Testing Samples  : {len(X_test)}")

        return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "employee_salary.csv"

    df = pd.read_csv(DATA_PATH)

    feature = FeatureEngineering(df)

    feature.drop_columns()

    feature.encode_features()

    X, y = feature.split_features_target()

    splitter = DataSplitter(X, y)

    X_train, X_test, y_train, y_test = splitter.split()

    print("\nTraining Features")
    print(X_train.head())

    print("\nTraining Target")
    print(y_train.head())