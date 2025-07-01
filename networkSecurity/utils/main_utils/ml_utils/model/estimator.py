import os,sys
from networkSecurity.constants.training_pipeline import MODEL_FILE_NAME, SAVED_MODEL_DIR

from networkSecurity.exception.exception import NetworkSecurityException

class NetworkModel:
    def __init__(self,preprocessor,model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def predict(self,x):
        try:
            x_transform = self.preprocessor.transform(x)
            y_pred = self.model.predict(x_transform)
            return y_pred
        except Exception as e:
            raise NetworkSecurityException(e, sys)