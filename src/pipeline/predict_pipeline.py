import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException



class CustomData:
    def __init__(self, gender, race_ethnicity, parental_level_of_education, lunch,
                 test_preparation_course, math_score, reading_score, writing_score, total_score):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.math_score = math_score
        self.reading_score = reading_score
        self.writing_score = writing_score
        self.total_score = total_score

    def get_data_as_DataFrame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "math_score": [self.math_score],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
                "total_score": [self.total_score]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)

    def predict_score(self, model, preprocessor, input_data):
        try:
            logging.info("Inside Scaling & Prediction pipeline")
            preprocessed_data = preprocessor.transform(input_data)
            predicted_scores = model.predict(preprocessed_data)
            return predicted_scores
        except Exception as e:
            raise CustomException(e, sys)
