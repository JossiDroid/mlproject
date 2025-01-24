import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
import sys 
from dataclasses import dataclass
import os 
from src.logger import logging
from src.exception import CustomException
from src.components.data_transformation import DataTransforamtion
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path :str = os.path.join('artifcats','train.csv')
    test_data_path :str = os.path.join('artifcats','test.csv')
    raw_data_path :str = os.path.join('artifcats','raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    logging.info("Entered the data Ingestion Module")
    def initiate_data_ingestion(self):
        try:
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info("Read the Data from the sheet")
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train Test Split Initated")

            train_set,test_set = train_test_split(df,test_size=0.25,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index =False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index =False,header=True)

            logging.info("Train Test data Completed ")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)


if __name__ =="__main__":
    obj = DataIngestion()
    train_path,test_path = obj.initiate_data_ingestion()

    transfomer_obj = DataTransforamtion()
    train_arr,test_arr,preprocessor_path= transfomer_obj.intiate_data_transformer(train_path=train_path,test_path=test_path)

    model_obj = ModelTrainer()
    print(model_obj.initiate_model_trainer(train_arr,test_arr,preprocessor_path))


    

