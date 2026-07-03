from pathlib import Path


# Project Root Directory
BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / "data" / "employees.csv"

# Data Folder
DATA_DIR = BASE_DIR / "data"

# Models Folder
MODELS_DIR = BASE_DIR / "models"

# Artifacts Folder
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Logs Folder
LOGS_DIR = BASE_DIR / "logs"


# Dataset
DATASET_PATH = DATA_DIR / "employee_salary.csv"

# Saved Models
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
LINEAR_MODEL_PATH = MODELS_DIR / "linear_regression.pkl"

