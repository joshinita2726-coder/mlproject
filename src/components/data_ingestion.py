import os
import sys
import pandas as pd

from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:

    train_data_path: str = os.path.join(
        "artifacts",
        "train.csv"
    )

    test_data_path: str = os.path.join(
        "artifacts",
        "test.csv"
    )

    raw_data_path: str = os.path.join(
        "artifacts",
        "data.csv"
    )


class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):

        logging.info(
            "Entered the data ingestion method or component"
        )

        try:

            # ------------------------------------------------
            # 1. Read the original dataset
            # ------------------------------------------------

            df = pd.read_csv(
                os.path.join(
                    "notebook",
                    "data",
                    "stud.csv"
                )
            )

            logging.info(
                "Read the dataset as dataframe"
            )

            # ------------------------------------------------
            # 2. Create artifacts directory
            # ------------------------------------------------

            os.makedirs(
                os.path.dirname(
                    self.ingestion_config.raw_data_path
                ),
                exist_ok=True
            )

            # ------------------------------------------------
            # 3. Save raw/original data
            # ------------------------------------------------

            df.to_csv(
                self.ingestion_config.raw_data_path,
                index=False,
                header=True
            )

            logging.info(
                "Raw data saved successfully"
            )

            # ------------------------------------------------
            # 4. Train-test split
            # ------------------------------------------------

            logging.info(
                "Train test split initiated"
            )

            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )

            # ------------------------------------------------
            # 5. Save training data
            # ------------------------------------------------

            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False,
                header=True
            )

            # ------------------------------------------------
            # 6. Save testing data
            # ------------------------------------------------

            test_set.to_csv(
                self.ingestion_config.test_data_path,
                index=False,
                header=True
            )

            logging.info(
                "Ingestion of data is completed"
            )

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:

            raise CustomException(e, sys)


# ------------------------------------------------------------
# Main execution
# ------------------------------------------------------------

if __name__ == "__main__":

    try:

        # Data Ingestion
        obj = DataIngestion()

        train_data, test_data = (
            obj.initiate_data_ingestion()
        )

        # Data Transformation
        data_transformation = DataTransformation()

        train_arr, test_arr, _ = (
            data_transformation.initiate_data_transformation(
                train_data,
                test_data
            )
        )

        # Model Training
        model_trainer = ModelTrainer()

        print(
            model_trainer.initiate_model_trainer(
                train_arr,
                test_arr
            )
        )

    except Exception as e:

        raise CustomException(e, sys)