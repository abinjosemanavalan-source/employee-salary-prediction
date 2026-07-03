import pandas as pd

from src.utils import ModelUtils


class PredictionPipeline:

    def __init__(self):

        self.model = ModelUtils.load_model("best_model.pkl")

    def predict(
        self,
        department,
        experience,
        education,
        age,
        gender,
        city
    ):

        # Encode categorical values
        department_map = {
            "Finance": 0,
            "HR": 1,
            "IT": 2,
            "Marketing": 3,
            "Sales": 4
        }

        education_map = {
            "Bachelor": 0,
            "Master": 1,
            "PhD": 2
        }

        gender_map = {
            "Female": 0,
            "Male": 1
        }

        city_map = {
            "Bangalore": 0,
            "Chennai": 1,
            "Delhi": 2,
            "Hyderabad": 3,
            "Mumbai": 4
        }

        data = pd.DataFrame([{
            "Department": department_map[department],
            "Experience_Years": experience,
            "Education_Level": education_map[education],
            "Age": age,
            "Gender": gender_map[gender],
            "City": city_map[city]
        }])

        prediction = self.model.predict(data)

        return prediction[0]


if __name__ == "__main__":

    pipeline = PredictionPipeline()

    salary = pipeline.predict(
        department="IT",
        experience=8,
        education="Bachelor",
        age=30,
        gender="Male",
        city="Bangalore"
    )

    print(f"\nPredicted Salary : ₹ {salary:.2f}")