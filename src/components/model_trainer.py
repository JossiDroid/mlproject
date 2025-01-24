import os 
import sys 
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging


from src.components.data_transformation import DataTransforamtion
from src.utils import save_obj,evaluate_model



@dataclass
class ModelTrainerConfig:
    tarined_model_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    def initiate_model_trainer(self,train_arr,test_arr,preprocessor_path):
        try:

            x_train,x_test,y_train,y_test =(train_arr[:,:-1],test_arr[:,:-1],train_arr[:,-1],test_arr[:,-1])

            models = {
                "RandomForest":RandomForestRegressor(),
                "DecisionTree":DecisionTreeRegressor(),
                "GradientBoosting":GradientBoostingRegressor(),
                "LinearRegression":LinearRegression(),
                "KNeighborsRegressor":KNeighborsRegressor(),
                "XGBRegressor":XGBRegressor(),
                "CatBoostRegressor":CatBoostRegressor(),
                "AdaBoostRegressor":AdaBoostRegressor(),
                "CatBoostRegressor":CatBoostRegressor(verbose=True) 
                }
            
            
            
            model_report:dict= evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)

            best_model_score = max(sorted(model_report.values()))


            best_model_name = [k for k,v in model_report.items() if v== best_model_score] 
            print(best_model_name)
            
            best_model = models[best_model_name[0]]

            logging.info("Best model Found")

            save_obj(file_path=self.model_trainer_config.tarined_model_path,
                     obj=best_model)
            
            logging.info("Best model saved")

            predicted = best_model.predict(x_test)
            r2score =r2_score(y_test,predicted)

            logging.info("Prediction and r2 score generated")

            return r2score
        except Exception as e:
            raise CustomException(e,sys)

