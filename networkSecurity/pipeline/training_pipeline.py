import os, sys

from networkSecurity.exception.exception import NetworkSecurityException
from networkSecurity.logging.logger import logging

from networkSecurity.components.data_ingestion import DataIngestion
from networkSecurity.components.data_transformation import DataTransformation
from networkSecurity.components.data_validation import DataValidation
from networkSecurity.components.model_trainer import ModelTrainer

from networkSecurity.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig, DataTransformationConfig, DataValidationConfig, ModelTrainerConfig
from networkSecurity.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifacts, DataValidationArtifact, ModelTrainerArtifacts

class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
    
    def data_ingestion(self):
        try:
            data_ingestion_config = DataIngestionConfig(self.training_pipeline_config)
            logging.info("Training PipeLine- Data Ingestion")
            data_ingestion = DataIngestion(data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            return data_ingestion_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def data_validation(self, data_ingestion_artifact: DataIngestionArtifact):
        try:
            data_validation_config = DataValidationConfig(self.training_pipeline_config)
            logging.info("Training Pipeline- Data Validation")
            data_validation = DataValidation(data_ingestion_artifact, data_validation_config)
            data_validation_artifact = data_validation.initiate_data_validation()
            return data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def data_transformation(self, data_validation_artifact: DataValidationArtifact):
        try:
            data_transformation_config = DataTransformationConfig(self.training_pipeline_config)
            logging.info("Training Pipeline- Data Transformation")
            data_transformation = DataTransformation(data_validation_artifacts=data_validation_artifact, data_transformation_config=data_transformation_config)
            data_transformation_artifact = data_transformation.inititate_data_transformation()
            return data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def model_trainer(self, data_transformation_artifact: DataTransformationArtifacts):
        try:
            model_trainer_config = ModelTrainerConfig(self.training_pipeline_config)
            logging.info("Training Pipeline- Model Trainer")
            model_trainer = ModelTrainer(model_trainer_config, data_transformation_artifact)
            model_trainer_artifacts = model_trainer.initiate_model_trainer()  
            return model_trainer_artifacts
    
        except Exception as e:
            raise NetworkSecurityException(e, sys)      
        
    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.data_ingestion()
            data_validation_artifact = self.data_validation(data_ingestion_artifact)
            data_transformation_artifact = self.data_transformation(data_validation_artifact)
            model_trainer_artifact = self.model_trainer(data_transformation_artifact)
            logging.info("Training PipeLine - Model Training Completed")
            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)