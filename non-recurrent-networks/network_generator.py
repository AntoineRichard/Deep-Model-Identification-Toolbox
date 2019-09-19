import models
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
        return models.GraphMLP_dX(settings, d, act=activation)

    elif name[0] == 'CNN':
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
        layer_type = []
        layer_param = []
        for di in d:
            if di[0] == 'p':
                layer_type.append('pool')
                layer_param.append(int(di[1]))
            elif di[0] == 'k':
                layer_type.append('conv')
                layer_param.append([int(di[1]),int(di[3:])])
            elif di[0] == 'd':
                layer_type.append('dense')
                layer_param.append(int(di[1:]))
        return models.GraphCNN_kXcX_pX_dX(settings, layer_type, layer_param, act=activation)

    else:
        raise('error unknown model type')
