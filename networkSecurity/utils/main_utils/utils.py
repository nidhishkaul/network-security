import yaml
from networkSecurity.exception.exception import NetworkSecurityException
from networkSecurity.logging.logger import logging
import os, sys
import dill
import numpy as np
import pickle
from sklearn.metrics import accuracy_score

def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise NetworkSecurityException(e, sys) 
    
def write_yaml_file(file_path: str, content: object, replace: bool=False):
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,"w") as file:
            yaml.dump(content,file)
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    
def save_numpy_array(file_path: str, array: np.array):
    """
    Save numpy array data to file
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path,"wb") as file:
            np.save(file, array)
    except Exception as e:
        raise NetworkSecurityException(e, sys)

def save_object(file_path: str, obj: object):
    """Save python objects to a pickle file."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file:
            pickle.dump(obj, file)
        logging.info("Object saved in the file.")
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    
def load_object(file_path: str):
    """Load python objects from a pickle file."""
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file path {file_path} does not exist.")
        with open(file_path,"rb") as file:
            obj_file = pickle.load(file)
            return obj_file
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    
def load_numpy_array(file_path: str):
    """Load numpy array data to a file"""
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file path {file_path} does not exist.")
        with open(file_path,"rb") as file:
            array = np.load(file=file)
            return array
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    
def evaluate_model(x_train, y_train, x_test, y_test, models:dict):
    try:
        report = {}
        trained_models = {}

        for name,model in models.items():
            classifier = model.fit(x_train, y_train)
            y_train_pred = classifier.predict(x_train)
            y_test_pred = classifier.predict(x_test)
            train_model_score = accuracy_score(y_train, y_train_pred)
            test_model_score = accuracy_score(y_test, y_test_pred)
            report[name] = {
                "train_accuracy":train_model_score,
                "test_accuracy":test_model_score
            }
            trained_models[name] = classifier

        best_model_name = max(report, key=lambda k: report[k]['test_accuracy'])
        best_model = trained_models[best_model_name]

        return best_model
    
    except Exception as e:
        raise NetworkSecurityException(e, sys)


