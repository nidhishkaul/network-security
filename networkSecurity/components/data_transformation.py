from networkSecurity.exception.exception import NetworkSecurityException
from networkSecurity.logging.logger import logging

from networkSecurity.entity.config_entity import DataTransformationConfig
from networkSecurity.entity.artifact_entity import DataTransformationArtifacts, DataValidationArtifact
 
import sys, os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
import pickle

from networkSecurity.constants.training_pipeline import TARGET_COLUMN
from networkSecurity.constants.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS
from networkSecurity.utils.main_utils.utils import save_numpy_array, save_object

class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig, data_validation_artifacts: DataValidationArtifact):
        try:
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifacts = data_validation_artifacts
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def get_data_transformer_object(cls)->Pipeline:
        """
        Initialise a KNN imputer with the specified params and return a pipeline with imputer as the first step.
        """
        logging.info("Creating the Data Transformation Pipeline.")
        try:
            imputer:KNNImputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            processor:Pipeline = Pipeline([
                ("imputer",imputer)
            ])
            return processor
        except Exception as e:
            raise NetworkSecurityException(e, sys)

        
    
    def inititate_data_transformation(self) -> DataTransformationArtifacts:
        logging.info("Data Transformation initiated")
        try:
            train_df = DataTransformation.read_data(self.data_validation_artifacts.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifacts.valid_test_file_path)

            # Training dataframe
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_train_df = target_feature_train_df.replace(-1,0)

            # Test dataframe
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            target_feature_test_df = target_feature_test_df.replace(-1,0)

            # Call Data Transformer Object
            preprocessor = self.get_data_transformer_object()
            preprocessor_obj = preprocessor.fit(input_feature_train_df)
            transformed_input_train = preprocessor_obj.transform(input_feature_train_df)
            transformed_input_test = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[transformed_input_train, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_test, np.array(target_feature_test_df)]

            # Save numpy array
            save_numpy_array(self.data_transformation_config.transformed_train_file_path, train_arr)
            save_numpy_array(self.data_transformation_config.transformed_test_file_path, test_arr)
            
            # Save preprocessor object
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor_obj)

            # Artifacts
            data_transformation_artifacts = DataTransformationArtifacts(
                transformed_object_file_path = self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path = self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path = self.data_transformation_config.transformed_test_file_path
            )

            return data_transformation_artifacts

        except Exception as e:
            raise NetworkSecurityException(e, sys)


