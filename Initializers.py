# encoding: utf-8
# env: py3.7, tf2.0
# author: ryan.Y
# tip: self defined initializers

#  if your idea is related to some modification on initialzation of parameters
#   please just pass this file to them. or copy the related classes

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.python.ops import init_ops_v2

class RandomNormal(tf.keras.initializers.Initializer):
    """Initializer that generates tensors with a normal distribution.
    Initializers allow you to pre-specify an initialization strategy, encoded in
    the Initializer object, without knowing the shape and dtype of the variable
    being initialized.
  
    Args:
        mean: a python scalar or a scalar tensor. Mean of the random values to
        generate.
        stddev: a python scalar or a scalar tensor. Standard deviation of the random
        values to generate.
        seed: A Python integer. Used to create random seeds. See
        `tf.random.set_seed` for behavior.
    """

    def __init__(self, mean=0.0, stddev=0.05, seed=None):
        self.mean = mean
        self.stddev = stddev
        self.seed = seed

    def __call__(self, shape, dtype=tf.dtypes.float32):
        """Returns a tensor object initialized as specified by the initializer.
        Args:
        shape: Shape of the tensor.
        dtype: Optional dtype of the tensor. Only floating point types are
         supported.
        Raises:
        ValueError: If the dtype is not floating point
        """
        
        dtype = _assert_float_dtype(dtype)
        tensorNp = np.random.normal(self.mean, self.stddev, shape)
        tensorTF = tf.convert_to_tensor(tensorNp,dtype=dtype)
        
        return tensorTF
    
    def get_config(self):
        return {
            "mean": self.mean,
            "stddev": self.stddev,
            "seed": self.seed
        }

class Initializer(object):
    """Initializer base class: all initializers inherit from this class.
    """
    def __call__(self, shape, dtype=None):
        """Returns a tensor object initialized as specified by the initializer.
            Args:
            shape: Shape of the tensor.
            dtype: Optional dtype of the tensor. If not provided will return tensor
            of `tf.float32`.
        """
        raise NotImplementedError

    def get_config(self):
        """Returns the configuration of the initializer as a JSON-serializable dict.
            Returns:
            A JSON-serializable Python dict.
        """
        return {}

    @classmethod
    def from_config(cls, config):
        """Instantiates an initializer from a configuration dictionary.
        Example:
        ```python
        initializer = RandomUniform(-1, 1)
        config = initializer.get_config()
        initializer = RandomUniform.from_config(config)
        ```
        Args:
            config: A Python dictionary.
            It will typically be the output of `get_config`.
        Returns:
            An Initializer instance.
        """
        config.pop("dtype", None)
        return cls(**config)

def _assert_float_dtype(dtype):
    """Validate and return floating point type based on `dtype`.
    `dtype` must be a floating point type.
    Args:
        dtype: The data type to validate.
    Returns:
        Validated type.
    Raises:
        ValueError: if `dtype` is not a floating point type.
    """
    dtype = dtypes.as_dtype(dtype)
    if not dtype.is_floating:
        raise ValueError("Expected floating point type, got %s." % dtype)
    return dtype
