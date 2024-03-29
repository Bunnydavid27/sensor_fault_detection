from datetime import datetime
import os
from sensor.constant import training_pipeline
from dataclasses import dataclass
from sensor.constant import prediction_pipeline
from sensor.constant.training_pipeline import MODEL_PUSHER_S3_KEY, MODEL_FILE_NAME
@dataclass
class TrainingPipelineConfig:

    def __init__(self, timestamp = datetime.now()) :
        timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.pipeline_name: str = training_pipeline.PIPELINE_NAME
        self.artifact_dir: str = os.path.join( training_pipeline.ARTIFACT_DIR, timestamp)
        self.timestamp: str = timestamp


@dataclass      
class DataIngestionConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        self.data_ingestion_dir: str = os.path.join(
        training_pipeline_config.artifact_dir, training_pipeline.DATA_INGESTION_DIR_NAME
        )

        self.feature_store_file_path: str = os.path.join(
            self.data_ingestion_dir, training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR, training_pipeline.FILE_NAME
        )

        self.training_file_path: str = os.path.join(
            self.data_ingestion_dir, training_pipeline.DATA_INGESTION_INGESTED_DIR, training_pipeline.TRAIN_FILE_NAME
        )
        
        self.testing_file_path: str = os.path.join(
            self.data_ingestion_dir, training_pipeline.DATA_INGESTION_INGESTED_DIR, training_pipeline.TEST_FILE_NAME
        )

        self.train_test_split_ratio: float = training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATION

        self.collection_name: str = training_pipeline.DATA_INGESTION_COLLECTION_NAME

        


@dataclass
class DataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig ):
        self.data_validation_dir: str = os.path.join(training_pipeline_config.artifact_dir, training_pipeline.DATA_VALIDATION_DIR_NAME)
        self.valid_data_dir: str = os.path.join(self.data_validation_dir,training_pipeline.DATA_VALIDATION_VALID_DIR)
        self.invalid_data_dir: str = os.path.join(self.data_validation_dir,training_pipeline.DATA_VALIDATION_INVALID_DIR)
        self.valid_train_file_path:str = os.path.join(self.valid_data_dir, training_pipeline.TRAIN_FILE_NAME )
        self.valid_test_file_path:str = os.path.join(self.valid_data_dir, training_pipeline.TEST_FILE_NAME)
        self.invalid_train_file_path:str = os.path.join(self.invalid_data_dir, training_pipeline.TRAIN_FILE_NAME )
        self.invalid_test_file_path:str = os.path.join(self.invalid_data_dir, training_pipeline.TEST_FILE_NAME )
        self.drift_report_file_path:str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME
            )
        


@dataclass
class DataTransformationConfig:
    def __init__(self,training_pipeline_config: TrainingPipelineConfig):
        self.data_transformation_dir: str = os.path.join(training_pipeline_config.artifact_dir, training_pipeline.DATA_TRANSFORMATION_DIR_NAME)

        self.transformed_train_file_path: str = os.path.join(self.data_transformation_dir,training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,training_pipeline.TRAIN_FILE_NAME.replace("csv","npy"))

        self.transformed_test_file_path: str = os.path.join(self.data_transformation_dir,training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,training_pipeline.TEST_FILE_NAME.replace("csv","npy"))

        self.transformed_object_file_path: str = os.path.join(self.data_transformation_dir, training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR, training_pipeline.PREPROCESSING_OBJECT_FILE_NAME)
        
        
        

@dataclass
class ModelTrainerConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.model_trainer_dir : str = os.path.join(training_pipeline_config.artifact_dir, training_pipeline.MODEL_TRAINER_DIR_NAME)
        self.trained_model_file_path: str = os.path.join(self.model_trainer_dir, training_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR)
        self.expected_accuracy:float = training_pipeline.MODEL_TRAINER_EXPECTED_SCORE
        self.overfitting_underfitting_threshold:float = training_pipeline.MODEL_TRAINER_OVERFITTING_UNDERFITTING_THRESHOLD
        self.predict_file : str = os.path.join(prediction_pipeline.PREDICT_TEST,prediction_pipeline.PREDICTION_INPUT_FILE_NAME)


@dataclass
class ModelEvaluationConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        self.model_evaluation_dir : str = os.path.join(training_pipeline_config.artifact_dir, training_pipeline.MODEL_EVALUATION_DIR_NAME)
        self.report_file_name : str = os.path.join(self.model_evaluation_dir, training_pipeline.MODEL_TRAINER_DIR_NAME)
        self.change_threshold = training_pipeline.MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE
        

@dataclass
class ModelPusherConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        self.model_evaluation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, training_pipeline.MODEL_PUSHER_DIR_NAME
        )

        self.model_file_path = os.path.join(
            self.model_evaluation_dir, training_pipeline.MODEL_FILE_NAME
        )
        timestamp = round(datetime.now().timestamp())
        
        self.saved_model_path = os.path.join(training_pipeline.SAVED_MODEL_DIR, f"{timestamp}", training_pipeline.MODEL_FILE_NAME)
        
@dataclass
class PredictionPipelineConfig:

    data_bucket_name: str = prediction_pipeline.PREDICTION_DATA_BUCKET

    data_file_path: str = prediction_pipeline.PREDICTION_INPUT_FILE_NAME

    model_file_path: str = os.path.join(MODEL_PUSHER_S3_KEY, MODEL_FILE_NAME)

    model_bucket_name: str = prediction_pipeline.MODEL_BUCKET_NAME

    output_file_name: str = prediction_pipeline.PREDICTION_OUTPUT_FILE_NAME

@dataclass
class Prediction_Config:

    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        timestamp = round(datetime.now().timestamp())
        self.prediction_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, prediction_pipeline.PREDICTION_DIR_NAME
        )

        self.prediction_file_path:str = os.path.join(
            self.prediction_dir, f"{timestamp}", prediction_pipeline.PREDICTION_OUTPUT_FILE_NAME
        )