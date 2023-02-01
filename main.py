from sensor.configuration.mongo_db_connection import MongoDBClient
from sensor.exception import SensorException
import os, sys
from sensor.logger import logging
from sensor.entity.config_entity import DataIngestionConfig, TrainingPipelineConfig
from sensor.pipeline.training_pipeline import TrainPipeline
"""
#if __name__ == '__main__':
    #mongodb_client = MongoDBClient()
    #print(mongodb_client.database.list_collection_names())

def test_exception():
    try:
        logging.info("We are dividing 1 by zero")
        x=1/0
    except Exception as e:
        raise SensorException(e,sys)

if __name__=='__main__':
    try:
        test_exception()
    except Exception as e:
        print(e)

"""
if __name__=='__main__':
    training_pipeline = TrainPipeline()
    training_pipeline.run_pipeline()


