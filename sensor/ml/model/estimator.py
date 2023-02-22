import sys

from pandas import DataFrame
from sklearn.pipeline import Pipeline

from sensor.exception import SensorException
from sensor.logger import logging


class TargetValueMapping:
    def __init__(self):
        self.neg: int = 0
        self.pos: int = 1

    
    def to_dict(self):
        return self.__dict__


    def reverse_mapping(self):
        mapping_response = self.to_dict()
        return dict(zip(mapping_response.values(), mapping_response.keys()))


class SensorModel:

    def __init__(self, preprocessor, model) -> None:
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise SensorException(e, sys)
        
    def predict(self, x):
        try:
            x_transform = self.preprocessor.transform(x)
            y_hat= self.model.predict(x_transform)
            return y_hat
            
        except Exception as e:
            raise SensorException(e,sys)