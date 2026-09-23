import sys
import os
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join(
        'artifacts',
        'preprocessor.pkl'
    )


class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        This function is responsible for data transformation.
        """

        try:
            numerical_columns = [
                'writing_score',
                'reading_score'
            ]

            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            # Numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent")
                    ),
                    (
                        "one_hot_encoder",
                        OneHotEncoder(handle_unknown='ignore')
                    )
                ]
            )

            logging.info(
                f"Categorical columns: {categorical_columns}"
            )

            logging.info(
                f"Numerical columns: {numerical_columns}"
            )

            # Combine both pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "num_pipeline",
                        num_pipeline,
                        numerical_columns
                    ),
                    (
                        "cat_pipeline",
                        cat_pipeline,
                        categorical_columns
                    )
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):

        try:
            # Read train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info(
                "Read train and test data completed"
            )

            # Get preprocessing object
            logging.info(
                "Obtaining preprocessing object"
            )

            preprocessing_obj = self.get_data_transformer_object()

            # Target column
            target_column_name = "math_score"

            # Separate input features and target
            input_feature_train_df = train_df.drop(
                columns=[target_column_name]
            )

            target_feature_train_df = train_df[
                target_column_name
            ]

            input_feature_test_df = test_df.drop(
                columns=[target_column_name]
            )

            target_feature_test_df = test_df[
                target_column_name
            ]

            logging.info(
                "Applying preprocessing object on training "
                "and testing dataframe"
            )

            # Fit and transform training data
            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )

            # Transform testing data
            input_feature_test_arr = preprocessing_obj.transform(
                input_feature_test_df
            )

            # Combine transformed features with target
            train_arr = np.c_[
                input_feature_train_arr,
                np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr,
                np.array(target_feature_test_df)
            ]

            logging.info(
                "Preprocessing completed successfully"
            )

            # Save preprocessing object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info(
                "Saved preprocessing object"
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)


def save_object(file_path, obj):
    """
    Save Python object as a pickle file.
    """

    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(
            dir_path,
            exist_ok=True
        )

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)