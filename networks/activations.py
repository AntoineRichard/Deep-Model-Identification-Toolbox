import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as Layers
import warnings 

'''
def evonorm_s0(x, gamma, beta, nonlinearity):
    if nonlinearity:
        v = trainable_variable_ones(shape=gamma.shape)
        num = x * tf.nn.sigmoid(v * x)
        return num / group_std(x) * gamma + beta
    else:
        return x * gamma + beta

def evonorm_b0(x, gamma, beta, nonlinearity, training):
    if nonlinearity:
        v = trainable variable ones(shape=gamma.shape)
        batch_std = batch_mean_and_std(x, training)
        den = tf.maximum(batch_std, v * x + instance_std(x))
        return x / den * gamma + beta
    else:
        return x * gamma + beta

def instance_std(x, eps=1e−5):
    var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
    return tf.sqrt(var + eps)

def group_std(x, groups=32, eps=1e−5):
    N, SS, IS = x.shape
    x = tf.reshape(x, [N, SS, groups, IS // groups])
    var = tf.nn.moments(x, [1, 2, 4], keepdims=True)
    std = tf.sqrt(var + eps)
    std = tf.broadcast_to(std, x.shape)
    return tf.reshape(std, [N, SS, IS])

def trainable_variable_ones(shape, name="v"):
   return tf.get_variable(name, shape=shape,initializer=tf.ones_initializer())
'''

def cswish(inputs):
    return tf.math.multiply(inputs, tf.nn.sigmoid(inputs))

class swish(Layers.Layer):
    def __init__(self):
        super(swish, self).__init__()
        self.beta = tf.Variable(1.0, trainable=True)

    def call(self, inputs):
        tf.summary.scalar('swish_value', self.beta)
        return tf.math.multiply(inputs,tf.math.multiply(self.beta, tf.nn.sigmoid(inputs)))

def mish(inputs):
    return tf.math.multiply(inputs,tf.nn.tanh(tf.nn.softplus(inputs)))

def get_activation(name):
    """
    Parses a string to extract the activation function.
    """
    string = name.split(':')

    if string[0] == 'RELU':
        activation = Layers.ReLU()
    elif string[0] == 'TANH':
        activation = Layers.Activation(tf.keras.activations.tanh)
    elif string[0] == 'SIGMOID':
        activation = Layers.Activation(tf.keras.activations.sigmoid)
    elif string[0] == 'LRELU':
        try:
            activation = Layers.LeakyReLU(float(string[1]))
        except:
            warnings.warn('Using default alpha parameter : 0.1.', SyntaxWarning)
            warnings.warn('To set a custom alpha value type LRELU:alpha instead of LRELU', SyntaxWarning)
            activation = Layers.LeakyReLU(0.1)
    elif string[0] == 'PRELU':
        activation = Layers.PReLU()
    elif string[0] == 'ELU':
        try:
            activation = Layers.ELU(float(string[1]))
        except:
            warnings.warn('Using default alpha parameter : 1.0.', SyntaxWarning)
            warnings.warn('To set a custom alpha value type LRELU:alpha instead of LRELU', SyntaxWarning)
            activation = Layers.ELU(1.0)
    elif string[0] == 'SELU':
        activation = Layers.Activation(tf.keras.activations.selu)
    elif string[0] == 'SWISH':
        activation = swish()
    elif string[0] == 'CSWISH':
        activation = Layers.Activation(cswish)
    elif string[0] == 'MISH':
        activation = Layers.Activation(mish)
    else:
        raise ValueError('error: unknown activation function. Currently supported activation functions are RELU, PRELU, LRELU, SELU, ELU, TANH, SIGMOIG, SWISH, CSWISH (constan version of swish), MISH.')
    return activation

def apply_optimizer(opt, learning_rate, loss):
    """
    Get the optimizer selected by the user
    """
    opt = opt.lower()
    if opt == 'adam':
        return tf.train.AdamOptimizer(learning_rate).minimize(loss)
    elif opt == 'sgd':
        return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    elif opt == 'momentum':
        return tf.train.MomentumOptimizer(learning_rate).minimize(loss)
    elif opt == 'adagrad':
        return tf.train.AdagradOptimizer(learning_rate).minimize(loss)
    elif opt == 'rmsprop':
        return tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
    else:
        raise ValueError('error: unknown optimizer. Currently supported optimizers are: adam, sgd, momentum, adagrad, rmsprop.')

def apply_decay(step, learning_rate, settings):
    """
    Applies decay is selected by user
    """
    mode = settings.decay_mode.lower()
    if mode == 'none':
        return learning_rate
    elif mode == 'exponential':
        return tf.train.exponential_decay(learning_rate, step, settings.decay_steps, settings.decay_rate)
    elif mode == 'polynomial':
        return tf.train.polynomial_decay(learning_rate, step, settings.decay_steps, settings.end_learning_rate, settings.decay_power)
    elif mode == 'inversetime':
        return tf.train.inverse_time_decay(learning_rate, step, settings.decay_steps, settings.decay_rate)
    elif mode == 'naturalexp':
        return tf.train.natural_exp_decay(learning_rate, step, settings.decay_steps, settings.decay_rate)
    elif mode == 'piecewiseconstant':
        return tf.train.piecewise_constant_decay(step, settings.decay_boundaries, settings.decay_values)
    elif mode == 'gamma':
        return learning_rate * tf.pow(settings.decay_rate, tf.floor(step/settings.decay_steps))
    else:
        raise ValueError('error: unknown decay type. Currently supported types are: none, exponential, polynomial, inversetime, naturalexp, piecewiseconstant, gamma.') 
