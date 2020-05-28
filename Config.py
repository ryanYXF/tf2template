# encoding: utf-8
# env: py3.7, tf2.0
# author: ryan.Y

# tip: Config may be recursively added to the parent config
#   config
#       dataset_config
#           batch_size
#           shuffle_buffer_size
#           ......
#       loss_config
#       model_config
#       ...... 
    
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K


class Config(object):
    """
    config is used to combine different hyper parameters in hierachical structure

    config
        dataset_config
            batch_size
            shuffle_buffer_size
            ......
        loss_config
        model_config
        ...... 
    config = Config()
    dataset_config = config(dataset_json_dict)
    config.dataset_config = dataset_config
    
    all config objects are instance of this class, the relation among them are determined by users
    """
    def __init__(self, dictionary={}):
        self.update(dictionary)
    @classmethod
    def fn_from_json_file(cls, json_file: str):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)
    def fn_to_json(self):
        pass
    def fn_update(self, dictionary):
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error("Can't set {} with value {} for {}".format(key, value, self))
                raise err
    
