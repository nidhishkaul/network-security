import yaml
from networkSecurity.exception.exception import NetworkSecurityException
from networkSecurity.logging.logger import logging
import os, sys
import dill
import numpy as np
import pickle

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