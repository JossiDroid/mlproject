import os 
import sys 

from src.exception import CustomException
from src.logger import logging

from sklearn.metrics import mean_absolute_error,root_mean_squared_error,r2_score

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
    
def evaluate_model(x_train,y_train,x_test,y_test,models):
    for k,v in models.items():
        md = v
        md.fit(x_train,y_train)

        y_pred = md.predict(x_test)

        report  = {
            f"{k}":r2_score(y_test,y_pred)

        }
    return report
        





