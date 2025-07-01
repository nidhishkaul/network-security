import os, sys
from networkSecurity.entity.artifact_entity import ClassificationMetricArtifact
from networkSecurity.exception.exception import NetworkSecurityException
from sklearn.metrics import f1_score, precision_score, accuracy_score

def get_classification_score(y_true,y_pred) -> ClassificationMetricArtifact:
    try:
        model_f1_score = f1_score(y_true,y_pred) 
        model_precision_score = precision_score(y_true,y_pred)
        model_accuracy_score = accuracy_score(y_true,y_pred)

        classification_mertic = ClassificationMetricArtifact(f1_score=model_f1_score, accuracy_score=model_accuracy_score, precision_score=model_precision_score)
        return classification_mertic
    except Exception as e:
        raise NetworkSecurityException(e, sys)