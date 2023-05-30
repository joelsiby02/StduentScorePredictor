import os
import sys
import pandas as pd
import numpy as np

from src.logger import logging
from src.exception import CustomException
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer

# from src.components.data_ingestion import DataIngestion,DataIngestionConfig
from src.utils import save_object



@dataclass
class DataTransformationConfig:
    preProcessor: str = os.path.join('artifacts', 'preprocessed.pkl')

class DataTransformation:
    def __init__(self):
        self.transform_config = DataTransformationConfig()
        
    def Obj_DataTransform(self):
        try:
            # Separating cat & num features
            cat_cols = ['gender', 'race_ethnicity', "parental_level_of_education", 'lunch', 'test_preparation_course']
            num_cols = ["writing_score", "reading_score"]
              
            # Adding new features
            # num_cols += ['total_score', 'average']

            
            # For Num features:
            numerical_pipeline = make_pipeline(
                SimpleImputer(strategy='median'),  # Impute missing values with median strategy
                StandardScaler()  # Scale the numerical features
            )

            # For Categorical features:
            categorical_pipeline = make_pipeline(
                SimpleImputer(strategy='most_frequent'),
                OneHotEncoder(),  # Perform one-hot encoding on categorical features
                # StandardScaler()  # Scale the categorical features
            )
            
            logging.info("Preprocessing done for Categorical & Numerical Features")
            
            # Column Transformer
            ct = ColumnTransformer([
                ('numerical', numerical_pipeline, num_cols),
                ('categorical', categorical_pipeline, cat_cols)
            ])
            
            logging.info("Return of Column Transformer Done~")
            return ct

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_DataTransformation(self, train_path, test_path):
        try:
            # Reading the dataFrames & logging
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Reading Train & Test Dataset Confirmed")
            
            # inheriting the properties from above class
            obj_inherit = self.Obj_DataTransform()
            
            # Specifying the target column
            target_col_name = 'math_score'
            
            # Separating features and target for train_df
            X_train = train_df.drop(columns=[target_col_name], axis=1)
            y_train = train_df[target_col_name]
            
            logging.info("Separating the input features and the target feature from the train datasets")

            # Separating features and target for test_df
            X_test = test_df.drop(columns=[target_col_name], axis=1)
            y_test = test_df[target_col_name]

            logging.info("Separating the input features and the target feature from  test datasets")
            
             
            input_feature_train_arr=obj_inherit.fit_transform(X_train)
            input_feature_test_arr=obj_inherit.transform(X_test)

            train_arr = np.c_[
                input_feature_train_arr, np.array(y_train)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(y_test)]

            
            
            
            
            
            
            
            
            
            
            # # Fitting and transforming on X_train
            # fit_transform_Xtrain_features = obj_inherit.fit_transform(X_train)
            # # Transforming X_test
            # transform_Xtest_features = obj_inherit.transform(X_test)
            
            # # Fitting and transforming on y_train
            # fit_transform_ytrain_features = obj_inherit.fit_transform(y_train)
            # # Transforming y_test
            # transform_ytest_features = obj_inherit.transform(y_test)
            
            # # Concatenating the transformed features
            # train_arr = np.c_[fit_transform_Xtrain_features, fit_transform_ytrain_features]
            # test_arr = np.c_[transform_Xtest_features, transform_ytest_features]
            
            # Logging completion
            logging.info("Data transformation completed")
            
            # save_object from utils.py
            save_object(
                file_path=self.transform_config.preProcessor, obj=obj_inherit
            )

            return train_arr, test_arr, self.transform_config.preProcessor,
        
        except Exception as e:
            raise CustomException(e, sys)

