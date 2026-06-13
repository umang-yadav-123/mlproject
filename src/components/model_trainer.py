import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

from src.exception import customException
from src.loggers import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join(
        "artifacts",
        "model.pkl"
    )


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing data")

            X_train, y_train = (
                train_array[:, :-1],
                train_array[:, -1]
            )

            X_test, y_test = (
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "KNN Regressor": KNeighborsRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "XGBoost Regressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False)
            }
            params = {
                "DecisionTree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "RandomForest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "GradientBoosting": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7,0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "LinearRegression": {},
                "KNeighborsRegressor": {
                    'n_neighbors': [5, 7, 9, 11],
                },
                "XGBRegressor": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoostRegressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoostRegressor": {
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }


            model_report = {}

            for model_name, model in models.items():
                model.fit(X_train, y_train)

                y_test_pred = model.predict(X_test)

                test_model_score = r2_score(y_test, y_test_pred)

                model_report[model_name] = test_model_score

            best_model_score = max(model_report.values())

            best_model_name = max(
                model_report,
                key=model_report.get
            )

            best_model = models[best_model_name]

            logging.info(
                f"Best model found: {best_model_name}"
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            prediction = best_model.predict(X_test)

            r2_square = r2_score(y_test, prediction)

            return r2_square

        except Exception as e:
            raise customException(e, sys)
        
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        model_report = {}

        for model_name, model in models.items():
            gs=GridSearchCV(
                estimator=model,
                param_grid=params[model_name],
                cv=3,
                n_jobs=-1,
                verbose=2
            )
            #model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)

            test_model_score = r2_score(y_test, y_test_pred)

            model_report[model_name] = test_model_score

        return model_report

    except Exception as e:
        raise customException(e, sys)