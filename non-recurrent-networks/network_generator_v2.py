import tf_graphv2
import tensorflow as tf

def get_graph(settings):
    name = settings.model.split('_')
    if name[0] == 'MLP':
        if name[1] == 'RELU':
            activation = tf.nn.relu
        elif name[1] == 'TANH':
            activation = tf.nn.tanh
        elif name[1] == 'SIGMOID':
            activation = tf.nn.sigmoid
        elif name[1] == 'LRELU':
            print('using LEAKY RELU')
            activation = tf.nn.leaky_relu
        else:
            raise('error: unknown activation function')
        d = name[2:]
        d = [int(x) for x in d]
        return tf_graphv2.GraphMLP_lX_X(settings, d, act=activation)

    elif name[0] == 'CNN':
        if name[2:]:
            print('Oh no...')
    else:
        raise('error unknown model type')
