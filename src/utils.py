import joblib
from pathlib import Path


class ModelUtils:
    """
    Utility class for saving and loading machine learning models.
    """

    @staticmethod
    def save_model(model, filename):

        models_dir = Path(__file__).resolve().parent.parent / "models"

        models_dir.mkdir(exist_ok=True)

        model_path = models_dir / filename

        joblib.dump(model, model_path)

        print(f"[OK] Model saved to: {model_path}")

    @staticmethod
    def load_model(filename):

        model_path = (
            Path(__file__).resolve().parent.parent
            / "models"
            / filename
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

        print(f"[OK] Model loaded from: {model_path}")

        return joblib.load(model_path)


# -------------------------
# Testing the utility
# -------------------------
if __name__ == "__main__":

    model = ModelUtils.load_model("best_model.pkl")

    print(type(model))