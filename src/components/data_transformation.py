from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
import os
from sklearn.impute import SimpleImputer
import pandas as pd 
import numpy as np 
from src.utils import save_obj
import sys

@dataclass
class DataTransformerConfig:
    preproccesor_path: str = os.path.join('artifcats','preprocessor.pkl')


class DataTransforamtion:
    def __init__(self):
        self.data_transformer_config = DataTransformerConfig()
    def prepoccesor_pipline(self):
        num_feature = [
            "writing_score","reading_score"
        ]
        cat_feature = [
            "gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"
        ]

        num_pipline  = Pipeline(
            steps=[("Imputer",SimpleImputer(strategy='median')),
                   ("Scalar",StandardScaler())]
        )

        cat_pipline = Pipeline(steps=[
            ("imputer",SimpleImputer(strategy='most_frequent')),
            ("oneHot",OneHotEncoder())
        ]

        )
        logging.info(f"Categorical feature is created")
        logging.info("Numerical Feature is Created")

        preprocessor = ColumnTransformer(
           [ ('num_pipline',num_pipline,num_feature),
            ("cat_pipline",cat_pipline,cat_feature)]
        )


        return preprocessor
    
    def intiate_data_transformer(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and Test Dataframe Created")

            target_feature = 'math_score'
            numerical_columns = ['writing_score','reading_score']

            input_features_train_df = train_df.drop(columns=target_feature)
            target_features_train_df = train_df[target_feature]

            input_features_test_df = test_df.drop(columns=target_feature)
            target_features_test_df = test_df[target_feature]

            logging.info("Created  Input and Target Features")

            preprocessor_obj = self.prepoccesor_pipline()

            input_features_train_df_arr=preprocessor_obj.fit_transform(input_features_train_df)
            input_features_test_df_arr = preprocessor_obj.transform(input_features_test_df)

            save_obj (
                file_path = self.data_transformer_config.preproccesor_path,
                obj = preprocessor_obj
            )

            train_arr = np.c_[
                input_features_train_df_arr,target_features_train_df
            ]

            test_arr = np.c_[
                input_features_test_df_arr,target_features_test_df
            ]
            logging.info("Train and Test Array is created successfully")

            return (
                train_arr,
                test_arr,
                self.data_transformer_config.preproccesor_path
            )

        except Exception as e:
            raise CustomException(e,sys)

