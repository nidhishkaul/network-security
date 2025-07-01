from networkSecurity.exception.exception import NetworkSecurityException
from networkSecurity.logging.logger import logging

from networkSecurity.entity.config_entity import ModelTrainerConfig
from networkSecurity.entity.artifact_entity import DataTransformationArtifacts, ModelTrainerArtifacts, ClassificationMetricArtifact

import sys, os

from networkSecurity.utils.main_utils.utils import save_object, load_object
from networkSecurity.utils.main_utils.utils import save_numpy_array, load_numpy_array, evaluate_model
from networkSecurity.utils.main_utils.ml_utils.mertic.classification_metric import get_classification_score
from networkSecurity.utils.main_utils.ml_utils.model.estimator import NetworkModel

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifacts: DataTransformationArtifacts):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifacts = data_transformation_artifacts
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def train_model(self, x_train, y_train, x_test, y_test):
        models = {
            "Random Forest":RandomForestClassifier(verbose=1),
            "AdaBoost": AdaBoostClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boost": GradientBoostingClassifier(verbose=1),
            "Logistic Regression": LogisticRegression(verbose=1)
        }

        # Get the best model
        model = evaluate_model(x_train, y_train, x_test, y_test, models)
        y_test_pred = model.predict(x_test)
        y_train_pred = model.predict(x_train)

        classification_metric_test = get_classification_score(y_test, y_test_pred)
        classification_metric_train = get_classification_score(y_train, y_train_pred)

        # Load the preprocessor
        preprocessor = load_object(self.data_transformation_artifacts.transformed_object_file_path)

        model_dir_path = os.path.dirname(self.model_trainer_config.trainer_model_file_path)
        os.makedirs(model_dir_path, exist_ok=True)

        network_model = NetworkModel(preprocessor=preprocessor, model=model)
        save_object(self.model_trainer_config.trainer_model_file_path, obj = network_model)

        # Model Trainer Artifacts
        model_trainer_artifact = ModelTrainerArtifacts(trainer_model_file_path=self.model_trainer_config.trainer_model_file_path,
            train_metric_artifact=classification_metric_train,
            test_metric_artifact=classification_metric_test,
        )

        return model_trainer_artifact
        
    def initiate_model_trainer(self)->ModelTrainerArtifacts:
        logging.info("Model Trainer Inititated")
        try:
            train_file_path = self.data_transformation_artifacts.transformed_train_file_path
            test_file_path = self.data_transformation_artifacts.transformed_test_file_path

            # Load train and test array
            train_arr = load_numpy_array(train_file_path)
            test_arr = load_numpy_array(test_file_path)

            x_train,y_train,x_test,y_test = (train_arr[:,:-1],train_arr[:,-1],test_arr[:,:-1],test_arr[:,-1])

            # Train Model
            logging.info("Model Training Started")
            model_trainer_artifact = self.train_model(x_train, y_train, x_test, y_test)
            logging.info("Model Training Completed")
            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)



