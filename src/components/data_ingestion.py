import os
import sys
from src.logger import logging
from src.exception import CustomException

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from data_transformation import DataTransformation, DataTransformationConfig


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")
    # preprocessed_path: str = os.path.join("artifacts", "preprocessed.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        logging.info("Initialized DataIngestion")

    def initiate_data_ingestion(self):
        try:
            # Load the DataFrame
            df = pd.read_csv('notebook\data\DATA_ORG.csv')
            logging.info("DataFrame loaded")

            # Create directories for train, raw, test data
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.test_data_path), exist_ok=True)
            logging.info("Directories created")

            # Split the raw data into train and test sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Train and test split done")

            # Save the raw, train, and test sets as CSV files
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Raw, Train, and test CSV files created")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.raw_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)



if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data, _ = obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_DataTransformation(train_data, test_data)
