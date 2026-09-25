import os
import sys

import dill

from src.exception import CustomException

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):

    try:

        dir_path = os.path.dirname(file_path)

        os.makedirs(
            dir_path,
            exist_ok=True
        )

        with open(
            file_path,
            "wb"
        ) as file_obj:

            dill.dump(
                obj,
                file_obj
            )

    except Exception as e:

        raise CustomException(e, sys)


def evaluate_models(
    X_train,
    y_train,
    X_test,
    y_test,
    models,
    param
):

    try:

        report = {}

        for model_name in models.keys():

            model = models[model_name]

            para = param[model_name]

            # Grid Search
            gs = GridSearchCV(
                model,
                para,
                cv=3,
                scoring="r2"
            )

            gs.fit(
                X_train,
                y_train
            )

            # Set best parameters
            model.set_params(
                **gs.best_params_
            )

            # Train model
            model.fit(
                X_train,
                y_train
            )

            # Training prediction
            y_train_pred = model.predict(
                X_train
            )

            # Testing prediction
            y_test_pred = model.predict(
                X_test
            )

            # Training R2 score
            train_model_score = r2_score(
                y_train,
                y_train_pred
            )

            # Testing R2 score
            test_model_score = r2_score(
                y_test,
                y_test_pred
            )

            report[model_name] = test_model_score

            print(
                f"{model_name}: "
                f"Train R2 = {train_model_score:.4f}, "
                f"Test R2 = {test_model_score:.4f}"
            )

        return report

    except Exception as e:

        raise CustomException(e, sys)