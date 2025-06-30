from networkSecurity.exception.exception import NetworkSecurityException
from networkSecurity.logging.logger import logging

from networkSecurity.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from networkSecurity.entity.config_entity import DataValidationConfig # Configuration for the data validation
from scipy.stats import ks_2samp 
import pandas as pd
import os,sys
from networkSecurity.constants.training_pipeline import SCHEMA_FILE_PATH
from networkSecurity.utils.main_utils.utils import read_yaml_file, write_yaml_file

class DataValidation:
    def __init__(self, data_ingestion_artifact:DataIngestionArtifact, data_validation_config:DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def validate_no_of_columns(self, dataframe:pd.DataFrame)->bool:
        try:
            no_of_columns = len(self._schema_config)
            logging.info(f"Required no of columns:{no_of_columns}")
            logging.info(f"Data Frame has columns:{len(dataframe.columns)}")
            if len(dataframe.columns) == no_of_columns:
                return True
            return False
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def detect_dataset_drift(self,base_df,current_df,threshold=0.05)->bool:
        try:
            status = True
            report = {}
            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]
                sample_dist = ks_2samp(d1,d2)
                if threshold<sample_dist.pvalue:
                    is_found = False
                else:
                    is_found = True
                    status = False
                report.update(
                    {column:{
                        "p_value":float(sample_dist.pvalue),
                        "drift_status":is_found
                    }}
                )
            drift_report_file_path = self.data_validation_config.drift_report_file_path
            write_yaml_file(drift_report_file_path, content=report) # Writing report in the yaml file

            # Create directory
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)

        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    
        
    def initiate_data_validation(self)->DataValidationArtifact:
        try:
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            # Read the data from train and test file
            train_df = DataValidation.read_data(train_file_path)
            test_df = DataValidation.read_data(test_file_path)

            # Validate number of columns and check for numeric columns
            status = self.validate_no_of_columns(train_df)
            if not status:
                error_message = f"Train DataFrame does not contain all columns.\n"            
            status = self.validate_no_of_columns(test_df)
            if not status:
                error_message = f"Test DataFrame does not contain all columns.\n"
            if not pd.api.types.is_numeric_dtype(train_df.columns):
                error_message = f"Train DataFrame contains non numerical columns.\n"
            if not pd.api.types.is_numeric_dtype(test_df.columns):
                error_message = f"Test DataFrame contains non numerical columns.\n"

            # Check DataDrift
            status = self.detect_dataset_drift(base_df=train_df,current_df=test_df)
            dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path, exist_ok=True)

            train_df.to_csv(self.data_validation_config.valid_train_file_path, index=False, header=True)
            test_df.to_csv(self.data_validation_config.valid_test_file_path, index=False, header=True)

            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_ingestion_artifact.trained_file_path,
                valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

            return data_validation_artifact
    
        except Exception as e:
            raise NetworkSecurityException(e, sys)

