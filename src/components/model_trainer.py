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
    tarined_model_path = os.path.join("artifcats","model.pkl")

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
            
            params={
                "DecisionTree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "RandomForest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "GradientBoosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "LinearRegression":{},
                "KNeighborsRegressor":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoostingRegressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoostRegressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
            
            
            
            model_report:dict= evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models,params=params)

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

