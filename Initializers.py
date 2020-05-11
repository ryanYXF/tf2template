# encoding: utf-8
#  py3.7, tf2.0
#  self defined initializers
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.python.ops import init_ops_v2


class RandomUniform(init_ops_v2.RandomUniform, keras.Initializer):
    """Initializer that generates tensors with a uniform distribution.

  Also available via the shortcut function
  `tf.keras.initializers.random_uniform`.

    Examples:

    >>> # Standalone usage:
    >>> initializer = tf.keras.initializers.RandomUniform(minval=0., maxval=1.)
    >>> values = initializer(shape=(2, 2))

    >>> # Usage in a Keras layer:
    >>> initializer = tf.keras.initializers.RandomUniform(minval=0., maxval=1.)
    >>> layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)

    Args:
        minval: A python scalar or a scalar tensor. Lower bound of the range of
        random values to generate (inclusive).
        maxval: A python scalar or a scalar tensor. Upper bound of the range of
        random values to generate (exclusive).
        seed: A Python integer. An initializer created with a given seed will
        always produce the same random tensor for a given shape and dtype.
    """

    def __call__(self, shape, dtype=None):
        """Returns a tensor object initialized as specified by the initializer.

        Args:
            shape: Shape of the tensor.
            dtype: Optional dtype of the tensor. Only floating point and integer
            types are supported. If not specified,
            `tf.keras.backend.floatx()` is used,
            which default to `float32` unless you configured it otherwise
            (via `tf.keras.backend.set_floatx(float_dtype)`).
        """
        return super(RandomUniform, self).__call__(shape, dtype=_get_dtype(dtype))