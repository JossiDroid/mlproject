import os 
import sys 

from src.exception import CustomException
from src.logger import logging

from sklearn.metrics import mean_absolute_error,root_mean_squared_error,r2_score
from sklearn.model_selection import GridSearchCV

import numpy as np 
import dill


def save_obj(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb')as file_obj:
            dill.dump(obj,file_obj)
    
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(x_train, y_train, x_test, y_test, models, params):
    report = {}
    for model_name, model in models.items():
        try:
            if model_name == "CatBoostRegressor":
                # CatBoost doesn't work with GridSearchCV
                model.set_params(**params[model_name][0])  # Set the first set of params manually
                model.fit(x_train, y_train)
            else:
                param = params.get(model_name, {})
                grid = GridSearchCV(model, param_grid=param, cv=3)
                grid.fit(x_train, y_train)
                
                model.set_params(**grid.best_params_)
                model.fit(x_train, y_train)

            y_pred = model.predict(x_test)
            report[model_name] = r2_score(y_test, y_pred)

        except Exception as e:
            print(f"Error with model {model_name}: {e}")
    return report

def load_object(file_path):
    try:
        with open(file_path,"rb")as file_obj:
            obj = dill.load(file_obj)
        return obj 
    
    except Exception as e:
        raise CustomException(e,sys)
        





