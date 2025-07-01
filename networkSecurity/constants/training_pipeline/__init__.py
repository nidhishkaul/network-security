import os
import sys
import numpy as np
import pandas as pd

"""
defining common constants for training pipeline
"""
TARGET_COLUMN = "Result"
PIPELINE_NAME: str = "NetworkSecurity"
ARTIFACT_DIR: str = "Artifacts"
FILE_NAME: str = "phisingData.csv"

TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"

SCHEMA_FILE_PATH = os.path.join("data_schema","schema.yaml")
SAVED_MODEL_DIR = os.path.join("saved_models")

"""
Data Ingestion related constants
"""
DATA_INGESTION_COLLECTION_NAME:str = "NetworkData"
DATA_INGESTION_DATABASE_NAME:str = "NIDAI"
DATA_INGESTION_DIRECTORY_NAME:str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR:str = "feature_store"
DATA_INGESTION_INGESTED_DIR:str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO:float = 0.2

"""
Data Validation related constants
"""
DATA_VALIDATION_DIRECTORY_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR : str = "validated"
DATA_VALIDATION_INVALID_DIR : str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR : str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME : str = "report.yaml"

"""
Data Transformation related constants
"""
DATA_TRANSFORMATION_DIRECTORY_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_objects"
PREPROCESSING_FILE_NAME: str = "preprocessing.pkl"

DATA_TRANSFORMATION_IMPUTER_PARAMS: dict = {"missing_values":np.nan,"n_neighbors":3,"weights":"uniform"} # KNN imputer to replace nan values

"""
Model Trainer related constants
"""
MODEL_TRAINER_DIRECTORY_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINER_MODEL_DIR: str = "trained_model"
MODEL_FILE_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_ACCURACY: float = 0.6
MODEL_TRAINER_OVERFITTING_UNDERFITTING_THRESHOLD: float = 0.05