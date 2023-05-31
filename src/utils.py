import pickle
import os
import sys
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def TrainModel(X_train, X_test, y_train, y_test, models):
    results = {}
    for model_name, model_info in models.items():
        model = model_info['model']
        params = model_info['params']

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Calculate evaluation metrics
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Generate report
        report = f"Model: {model_name}\n"
        report += f"RMSE: {rmse}\n"
        report += f"MAE: {mae}\n"
        report += f"R^2 Score: {r2}\n"
        report += "\n"

        results[model_name] = rmse

        # Save the report to a text file
        save_report_to_file(report, model_name)

        # Log the report
        logging.info(f"Report for {model_name}:\n{report}")

    return results


def save_report_to_file(report, model_name):
    file_path = f"model_reports/{model_name}_report.txt"
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "w") as file_obj:
            file_obj.write(report)

        logging.info(f"Report for {model_name} saved to {file_path}")

    except Exception as e:
        raise CustomException(e, sys)


def select_best_model(models, results):
    best_model = None
    best_rmse = np.inf
    
    for model_name, rmse in results.items():
        if rmse < best_rmse:
            best_model = models[model_name]['model']
            best_rmse = rmse
    logging.info(f"Selecting {best_model}")
    return best_model



...

def load_model_and_preprocessor(model_path="artifacts\Model.pkl", preprocessor_path="artifacts\preprocessed.pkl"):
    try:
        logging.info(f"Inside model & Prepocessed path in utils")
        with open(model_path, "rb") as model_file, open(preprocessor_path, "rb") as preprocessor_file:
            model = pickle.load(model_file)
            preprocessor = pickle.load(preprocessor_file)
        return model, preprocessor
    except Exception as e:
        raise CustomException(e, sys)



