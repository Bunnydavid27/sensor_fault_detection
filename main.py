from sensor.configuration.mongo_db_connection import MongoDBClient
from sensor.exception import SensorException
import os,sys
from sensor.logger import logging
from sensor.pipeline import training_pipeline
from sensor.pipeline.training_pipeline import TrainPipeline
import os
from fastapi.responses import ORJSONResponse
from sensor.constant.training_pipeline import SCHEME_FILE_PATH
from typing import Annotated
from datetime import datetime
from sensor.utils.main_utils import load_object
from sensor.utils.main_utils import read_yaml_file
from sensor.constant.training_pipeline import SAVED_MODEL_DIR, MONGO_DB_ENV_FILE_PATH
from sensor.constant import prediction_pipeline
from fastapi import FastAPI, File, UploadFile
from sensor.constant.application import APP_HOST, APP_PORT
from starlette.responses import RedirectResponse
from uvicorn import run as app_run
from fastapi.responses import Response
from sensor.ml.model.estimator import ModelResolver,TargetValueMapping
from sensor.utils.main_utils import load_object
from fastapi.middleware.cors import CORSMiddleware
from sensor.pipeline.prediction_pipeline import PredictionPipeline
import pandas as pd
from sensor.entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact, DataValidationArtifact
from sensor.utils.main_utils import load_numpy_array_data
from sensor.entity.config_entity import Prediction_Config, ModelTrainerConfig
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, Form
import csv
import codecs
import requests

env_file_path=MONGO_DB_ENV_FILE_PATH

from fastapi import File, UploadFile


def set_env_variable(env_file_path):
    env_config = read_yaml_file(env_file_path)
    os.environ['MONGO_DB_URL']=env_config['MONGO_DB_URL']

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags = ["authentication"])
async def index():
    return RedirectResponse(url="/docs")

# @app.post("/upload")
# def upload( files: UploadFile = File(...)):

#     return {"file": files} 

# @app.post("/upload")
# def upload(file:UploadFile):
#     try:
#         contents = file.file.read()
#         return contents
#         # with open(file.filename, 'wb') as f:
#         #     f.write(contents)
#     except Exception:
#         return {"message": "There was an error uploading the file"}
#     finally:
#         file.file.close()

#     return {"message": f"Successfully uploaded {file.filename}"}

@app.get("/train")
async def train_route():
   try:
        train_pipeline =TrainPipeline()
        if train_pipeline.is_pipeline_running:
            return Response("Training pipeline is already running")
        training_pipeline = TrainPipeline()
        training_pipeline.run_pipeline()
        return Response("Training SUccessful")
   except Exception as e:
        return Response(f"Error Occured{e}")
   
@app.get("/imp_details")  
async def root():
    return {1: "Data contains class",2:"Data Directly for prediction"}

@app.get("/predict")
async def predict_route(dataset: Annotated[str, Form()]):
    try:
        #get data from user csv file
        #conver csv file to dataframe
        # prediction_pipeline = PredictionPipeline()
        # prediction_pipeline.initiate_prediction()
        data = {"dataset": dataset}
        df_file = str(data.get('dataset'))
        print(df_file)  
        # df_file = create_upload_file(UploadFile)
        # print(df_file)
        # test_file_path = os.path.join(prediction_pipeline.PREDICT_TEST,prediction_pipeline.PREDICTION_INPUT_FILE_NAME) 
        # print(test_file_path)
        # df = load_object(test_file_path)
        # df = pd.DataFrame(df)
        df=pd.read_csv(df_file)
        df = df.drop('class',axis='columns')
        schema_config = read_yaml_file(SCHEME_FILE_PATH)
        df = df.drop(
                schema_config["drop_columns"], axis=1
            )
        print(df)
        model_resolver = ModelResolver(model_dir=SAVED_MODEL_DIR)
        if not model_resolver.is_model_exists():
            return Response("Model is not available")
        
        # best_model_path = model_resolver.get_best_model_path()
        # model = load_object(file_path=best_model_path)
        model = load_object(file_path=os.path.join(r"saved_models\1679902003\model.pkl"))
        y_pred = model.predict(df)
        y_pre = pd.DataFrame(y_pred)
        df['predicted_column'] = y_pred
        df['predicted_column'].replace(TargetValueMapping().reverse_mapping(),inplace=True)
        file_path_location = os.path.join(
            prediction_pipeline.PREDICT_TEST, prediction_pipeline.PREDICTION_OUTPUT_FILE_NAME
        )
        y_pred_file_path = os.path.join(
            prediction_pipeline.PREDICT_TEST, "y_pre.csv")
        df.to_csv(file_path_location)
        y_pre.to_csv(y_pred_file_path)
        
    except Exception as e:
        raise SensorException(e,sys)

def main():
    try:
        set_env_variable(env_file_path)
        training_pipeline = TrainPipeline()
        training_pipeline.run_pipeline()
    except Exception as e:
        print(e)
        logging.exception(e)


if __name__=="__main__":
    main()
    set_env_variable(env_file_path)
    app_run(app, host=APP_HOST, port=APP_PORT)


# if __name__=='__main__':
#     try:
#         env_file_path = "/config/workspace/env.yaml"
#         set_env_variable(env_file_path=env_file_path)
#         training_pipeline = TrainPipeline()
#         training_pipeline.run_pipeline()
#     except Exception as e:
#         raise SensorException(e,sys)


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