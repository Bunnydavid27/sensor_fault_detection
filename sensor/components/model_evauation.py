from sensor.exception import SensorException
from sensor.logger import logging
from sensor.entity.artifact_entity import ModelTrainerArtifact,DataValidationArtifact, DataTransformationArtifact
from sensor.entity.config_entity import ModelEvaluationConfig
import os, sys
from xgboost import XGBClassifier
from sensor.ml.metric.classification_metric import get_classification_score
from sensor.ml.model.estimator import SensorModel
from sensor.utils.main_utils import save_object, load_object


class ModelEvaluation:
    def __init__(self, model_eval_config:ModelEvaluationConfig,
                 data_validation_artifact: DataValidationArtifact,
                 model_trainer_artifact:ModelTrainerArtifact
                 ):
        
        try:
            pass
        except Exception as e:
            raise SensorException(e,sys)
