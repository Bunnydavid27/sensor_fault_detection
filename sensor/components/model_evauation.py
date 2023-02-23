from sensor.exception import SensorException
from sensor.logger import logging
from sensor.entity.artifact_entity import ModelTrainerArtifact,DataValidationArtifact,ModelEvaluationArtifact, DataTransformationArtifact
from sensor.entity.config_entity import ModelEvaluationConfig
import os, sys
from xgboost import XGBClassifier
from sensor.ml.metric.classification_metric import get_classification_score
from sensor.ml.model.estimator import SensorModel, ModelResolver
from sensor.utils.main_utils import save_object, load_object
import pandas as pd


class ModelEvaluation:
    def __init__(self, model_eval_config:ModelEvaluationConfig,
                 data_validation_artifact: DataValidationArtifact,
                 model_trainer_artifact:ModelTrainerArtifact
                 ):
        
        try:
            self.model_eval_config = model_eval_config
            self.data_validation_artifact = data_validation_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise SensorException(e,sys)

    def initiate_model_evaluation(self)->ModelEvaluationArtifact:
        try:
            valid_train_file_path  = self.data_validation_artifact.valid_train_file_path
            valid_test_file_path  = self.data_validation_artifact.valid_test_file_path

            train_df = pd.read_csv(valid_train_file_path)
            test_df = pd.read_csv(valid_test_file_path)

            model_resolver = ModelResolver()
            model_file_path = self.model_trainer_artifact.trained_model_file_path
            if not model_resolver.is_model_exists();
                model_evaluation_artifact= ModelEvaluationArtifact(is_model_accepted=is_model_accepted,
                                        improved_accuracy=None,
                                        best_model_path=None,
                                        trained_model_path=model_file_path,
                                        train_model_metric_artifact=self.model_trainer_artifact,
                                        best_model_metric_artifact=None)
                logging.info(f"Model evaluation artifact:{model_evaluation_artifact}")
                return model_evaluation_artifact
            
            latest_model_path =model_resolver.get_best_model_path()
            latest_model = load_object(file_path=latest_model_path)

            model_file_path = self.model_trainer_artifact.trained_model_file_path
            model = load_object(file_path=model_file_path)
