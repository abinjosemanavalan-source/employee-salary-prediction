import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder


class FeatureEngineering:
    """
    Handles feature engineering tasks:
    1. Remove unnecessary columns
    2. Encode categorical features
    3. Split features and target
    """

    def __init__(self, dataframe):
        self.df = dataframe

    def drop_columns(self):
        """Remove columns that are not useful for prediction."""

        columns_to_drop = ["EmployeeID", "Name"]

        self.df.drop(columns=columns_to_drop, inplace=True)

        print("✅ Unnecessary columns removed.")

        return self.df

    def encode_features(self):
        """Convert categorical columns into numeric values."""

        encoder = LabelEncoder()

        categorical_columns = [
            "Department",
            "Education_Level",
            "Gender",
            "City"
        ]

        for column in categorical_columns:
            self.df[column] = encoder.fit_transform(self.df[column])

        print("✅ Categorical columns encoded.")

        return self.df

    def split_features_target(self):
        """Split dataset into Features (X) and Target (y)."""

        X = self.df.drop("Monthly_Salary", axis=1)
        y = self.df["Monthly_Salary"]

        return X, y


if __name__ == "__main__":

    # Project root/data/employee_salary.csv
    DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "employee_salary.csv"

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Create object
    feature = FeatureEngineering(df)

    # Perform feature engineering
    feature.drop_columns()
    feature.encode_features()

    # Split data
    X, y = feature.split_features_target()

    print("\n========== FEATURES ==========\n")
    print(X.head())

    print("\n========== TARGET ==========\n")
    print(y.head())