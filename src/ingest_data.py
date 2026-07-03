import pandas as pd
import os


class DataIngestion:

    def __init__(self, filepath):
        self.filepath = filepath

    def load_data(self):

        if not os.path.exists(self.filepath):
            raise FileNotFoundError(
                f"Dataset not found: {self.filepath}"
            )

        df = pd.read_csv(self.filepath)

        print("=" * 50)
        print("Dataset Loaded Successfully")
        print("=" * 50)

        print("\nFirst Five Rows\n")
        print(df.head())

        print("\nShape")
        print(df.shape)

        print("\nColumns")
        print(df.columns.tolist())

        return df


if __name__ == "__main__":

    dataset = DataIngestion(
        "data/employee_salary.csv"
    )

    df = dataset.load_data()