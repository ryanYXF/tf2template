# encoding: utf-8
# env: py3.7, tf2.0
# author: ryan.Y 
# tip: code for optimizer

# the loss may be strong related to some parameters, 
# in this case, please prepare a tf.keras.layers.layer
# the tf.keras.losses.Loss class should not contain any trainable parameters, except hyperparameter

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from keras import initializers, activations
from keras.layers import *

class AMSoftmaxLayer(tf.keras.layers.Layer):
    """
    loss layer compatiable with AMSoftmaxLoss class
    this layer contains the parameters and specific operations for amsoftmax loss

    Arguments:
        tf {[type]} -- [description]
    """
    def __init__(self, hidden_dim,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs
                 ):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(AMSoftmax, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(shape=(input_dim, self.hidden_dim),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        #self.built = True

    def call(self, inputs, **kwargs):
        inputs = tf.nn.l2_normalize(inputs, dim=-1)
        self.kernel = tf.nn.l2_normalize(self.kernel, dim=(0, 1))   
        cosine = tf.matmul(inputs, self.kernel)
        output = tf.clip_by_value(cosine, -1, 1)
        
        return output

class AMSoftmaxLoss(tf.keras.losses.Loss):
    """
    the call interface is preserved, all related trainable parameters and specific operation before call interface
    should be implemented in the class  AMSoftmaxLayer
    the y_pred arg in call interface should be the output of AMSoftmaxLayer()

    Arguments:
        s {int} -- [hyper parameter for am softmax, default 30]
        s {float} -- [hyper parameter for am softmax, default 0.1]
    """
    def __init__(s=30, m=0.1,**kwargs):
        self.s = s
        self.m = m
    def call(y_true, y_pred):
        return amsoftmax_loss(y_true, y_pred, self.m, self.s)
    

def amsoftmax_loss(y_true, y_logits, m, s):
    y_pred = tf.where(tf.equal(y_true, 1), y_logits - m, y_logits)
    y_pred *= s
    am_loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        labels=y_true,
                        logits=y_pred
                    )
                )
    return am_loss