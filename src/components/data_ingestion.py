import os
import sys

from src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig


@dataclass
class DataIngestionConfig:

    data_path: str = os.path.join('artifacts', "data.csv")
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")


class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):

        logging.info("Entered the data ingestion method or component")

        try:

            # Read original dataset
            df = pd.read_csv(
                os.path.join('notebook', 'data', 'stud.csv')
            )

            logging.info("Read the dataset as dataframe")

            # Create artifacts folder
            os.makedirs(
                os.path.dirname(
                    self.ingestion_config.data_path
                ),
                exist_ok=True
            )

            # Save original dataset
            df.to_csv(
                self.ingestion_config.data_path,
                index=False,
                header=True
            )

            logging.info("Original dataset saved in artifacts")

            # Train-test split
            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )

            # Save train data
            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False,
                header=True
            )

            # Save test data
            test_set.to_csv(
                self.ingestion_config.test_data_path,
                index=False,
                header=True
            )

            logging.info("Ingestion of data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:

            raise CustomException(e, sys)


if __name__ == "__main__":

    obj = DataIngestion()

    train_data,test_data=obj.initiate_data_ingestion()

    data_tansformation=DataTransformation()
    data_tansformation.initiate_data_transformation(train_data,test_data)