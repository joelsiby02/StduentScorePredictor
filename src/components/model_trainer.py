import os
import sys
import pandas as pd

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, TrainModel, select_best_model
from dataclasses import dataclass

from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from src.logger import logging
from src.exception import CustomException
from data_transformation import DataTransformationConfig


@dataclass
class ModelTrainingConfig:
    Model_path: str = os.path.join("artifacts", "Model.pkl")


class ModelTraining():
    def __init__(self):
        self.Model_config = ModelTrainingConfig()

    def initiate_TrainModel(self, train_arr, test_arr):
        try:
            logging.info("Splitting Training & Test Array")

            X_train, X_test, y_train, y_test = (
                train_arr[:, :-1],
                test_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, -1]
            )
            models = {
                'linear_regression': {
                    'model': LinearRegression(),
                    'params': {}
                },
                'random_forest_regressor': {
                    'model': RandomForestRegressor(),
                    'params': {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [None, 5, 10],
                        'min_samples_split': [2, 5, 10]
                    }
                },
                'ridge_regression': {
                    'model': Ridge(),
                    'params': {
                        'alpha': [0.1, 1.0, 10.0],
                        'solver': ['auto', 'svd', 'cholesky']
                    }
                },
                'support_vector_regression': {
                    'model': SVR(),
                    'params': {
                        'C': [0.1, 1.0, 10.0],
                        'kernel': ['linear', 'rbf', 'poly']
                    }
                },
                'decision_tree_regressor': {
                    'model': DecisionTreeRegressor(),
                    'params': {
                        'max_depth': [None, 5, 10],
                        'min_samples_split': [2, 5, 10]
                    }
                },
                'xgboost_regressor': {
                    'model': XGBRegressor(),
                    'params': {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [3, 5, 7],
                        'learning_rate': [0.1, 0.01, 0.001]
                    }
                }
            }

            results = TrainModel(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, models=models)
            best_model = select_best_model(models=models, results=results)
            save_object(self.Model_config.Model_path, best_model)

        except Exception as e:
            raise CustomException(e, sys)
