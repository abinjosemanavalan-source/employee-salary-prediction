import pandas as pd
from pathlib import Path


class DataPreprocessor:

    def __init__(self, dataframe):
        self.df = dataframe

    def dataset_shape(self):
        print("\nDataset Shape")
        print("-" * 40)
        print(self.df.shape)

    def check_datatypes(self):
        print("\nData Types")
        print("-" * 40)
        print(self.df.dtypes)

    def check_missing_values(self):
        print("\nMissing Values")
        print("-" * 40)
        print(self.df.isnull().sum())

    def check_duplicates(self):
        print("\nDuplicate Rows")
        print("-" * 40)
        print(self.df.duplicated().sum())

    def remove_duplicates(self):
        before = self.df.shape[0]
        self.df = self.df.drop_duplicates()
        after = self.df.shape[0]

        print(f"\nRemoved {before - after} duplicate rows.")
        return self.df


if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_PATH = BASE_DIR / "data" / "employee_salary.csv"

    print("Reading:", DATA_PATH)

    df = pd.read_csv(DATA_PATH)

    processor = DataPreprocessor(df)

    processor.dataset_shape()
    processor.check_datatypes()
    processor.check_missing_values()
    processor.check_duplicates()
    processor.remove_duplicates()