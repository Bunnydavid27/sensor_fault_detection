import os
from sensor.constant.s3_bucket import TRAINING_BUCKET_NAME

SAVED_MODEL_DIR = os.path.join("saved_models")
MONGO_DB_ENV_FILE_PATH="config/workspace/env.yaml"

TARGET_COLUMN = "class"
PIPELINE_NAME: str = "sensor"
ARTIFACT_DIR: str = "artifact"
FILE_NAME: str = "sensor.csv"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"
MODEL_FILE_NAME = "model.pkl"
SCHEME_FILE_PATH = os.path.join("config","schema.yaml")
SCHEMA_DROP_COLS = "drop_columns"

"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_COLLECTION_NAME: str = "sensor"

DATA_INGESTION_DIR_NAME: str = "data_ingestion"

DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"

DATA_INGESTION_INGESTED_DIR: str = "ingested"

DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float = 0.2

"""
Data Validation realted contant start with DATA_VALIDATION VAR NAME
"""

DATA_VALIDATION_DIR_NAME: str = "data_validation"

DATA_VALIDATION_VALID_DIR: str = "validated"

DATA_VALIDATION_INVALID_DIR: str = "invalid"

DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"

DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"

"""
Data Transformation ralated constant start with DATA_TRANSFORMATION VAR NAME
"""

DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"

DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"

DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

"""
MODEL TRAINER related constant start with MODEL_TRAINER var name
"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer"

MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"

MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"

MODEL_TRAINER_EXPECTED_SCORE: float = 0.6

MODEL_TRAINER_OVERFITTING_UNDERFITTING_THRESHOLD: float = 0.05

MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join("config", "model.yaml")
"""
MODEL Evauation related constant start with MODEL_EVALUATION var name
"""

MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02

MODEL_EVALUATION_DIR_NAME:str = "model_evaluation"

MODEL_EVALUATION_REPORT_FILE_PATH = "report.yaml"


"""
MODEL PUSHER related constants that are used in MODEL_PUSHER
"""

MODEL_PUSHER_DIR_NAME = "model_pusher"

MODEL_PUSHER_BUCKET_NAME = TRAINING_BUCKET_NAME

MODEL_PUSHER_S3_KEY = "model-registry"
