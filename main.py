from networkSecurity.components.data_ingestion import DataIngestion
from networkSecurity.exception.exception import NetworkSecurityException
from networkSecurity.logging.logger import logging
from networkSecurity.entity.config_entity import DataIngestionConfig
from networkSecurity.entity.config_entity import TrainingPipelineConfig
import sys

if __name__ == '__main__':
    try:
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        logging.info("Initiate Data Ingestion")
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        print(data_ingestion_artifact)
    except Exception as e:
        raise NetworkSecurityException(e, sys)