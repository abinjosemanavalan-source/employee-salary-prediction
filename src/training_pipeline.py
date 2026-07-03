import pandas as pd
import sys

from src.config import DATASET_PATH
from src.logger import logger
from src.exception import ProjectException

from src.feature_engineering import FeatureEngineering
from src.data_splitter import DataSplitter
from save_best_model import BestModelTrainer


class TrainingPipeline:

    def run_pipeline(self):

        try:

            logger.info("Loading dataset...")

            df = pd.read_csv(DATASET_PATH)

            logger.info("Dataset loaded successfully.")

            feature = FeatureEngineering(df)

            feature.drop_columns()

            feature.encode_features()

            X, y = feature.split_features_target()

            logger.info("Feature engineering completed.")

            splitter = DataSplitter(X, y)

            X_train, X_test, y_train, y_test = splitter.split()

            logger.info("Train/Test split completed.")

            trainer = BestModelTrainer(
                X_train,
                X_test,
                y_train,
                y_test
            )

            trainer.train_and_save()

            logger.info("Training pipeline completed successfully.")

            print("\n🏆 Training Pipeline Finished Successfully!")

        except Exception as e:

            logger.error(str(e))

            raise ProjectException(e, sys)


if __name__ == "__main__":

    pipeline = TrainingPipeline()

    pipeline.run_pipeline()