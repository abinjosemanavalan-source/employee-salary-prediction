import joblib
import pandas as pd
from pathlib import Path


class SalaryPredictor:

    def __init__(self):

        model_path = Path(__file__).resolve().parent.parent / "models" / "linear_regression.pkl"

        self.model = joblib.load(model_path)

    def predict(self, employee_data):

        employee_df = pd.DataFrame([employee_data])

        prediction = self.model.predict(employee_df)

        return prediction[0]


if __name__ == "__main__":

    predictor = SalaryPredictor()

    employee = {
        "Department": 3,
        "Experience_Years": 10,
        "Education_Level": 2,
        "Age": 30,
        "Gender": 0,
        "City": 1
    }

    salary = predictor.predict(employee)

    print("\nPredicted Monthly Salary")
    print(f"₹ {salary:.2f}")