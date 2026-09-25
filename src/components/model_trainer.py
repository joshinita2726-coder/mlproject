import os
import sys

from dataclasses import dataclass

from catboost import CatBoostRegressor

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join(
        "artifacts",
        "model.pkl"
    )


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):

        try:

            logging.info(
                "Splitting training and test input data"
            )

            # Split input features and target variable
            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            # Define models
            models = {

                "Random Forest": RandomForestRegressor(),

                "Decision Tree": DecisionTreeRegressor(),

                "Gradient Boosting": GradientBoostingRegressor(),

                "Linear Regression": LinearRegression(),

                "K-Neighbors": KNeighborsRegressor(),

                "XGBoost": XGBRegressor(),

                "CatBoosting": CatBoostRegressor(verbose=False),

                "AdaBoost": AdaBoostRegressor()

            }

            # Evaluate all models
            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models
            )

            # Get best model score
            best_model_score = max(
                sorted(model_report.values())
            )

            # Get best model name
            best_model_name = list(
                model_report.keys()
            )[
                list(model_report.values()).index(
                    best_model_score
                )
            ]

            # Get best model
            best_model = models[best_model_name]

            logging.info(
                f"Best model found: {best_model_name}"
            )

            logging.info(
                f"Best model R2 score: {best_model_score}"
            )

            # Check model performance
            if best_model_score < 0.6:

                raise CustomException(
                    "No best model found with R2 score above 0.6"
                )

            # Save best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Prediction
            predicted = best_model.predict(X_test)

            # R2 score
            r2_square = r2_score(
                y_test,
                predicted
            )

            logging.info(
                f"Final R2 score: {r2_square}"
            )

            return r2_square

        except Exception as e:

            raise CustomException(e, sys)