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
            activation = tf.nn.leaky_relu
        else:
            raise('error: unknown activation function')
        d = name[2:]
        d = [int(x) for x in d]
        print('### USING MLP MODEL ###')
        return models.GraphMLP_dX(settings, d, act=activation)
    
    if name[0] == 'MLPCPLX':
        if name[1] == 'RELU':
            activation = tf.nn.relu
        elif name[1] == 'TANH':
            activation = tf.nn.tanh
        elif name[1] == 'SIGMOID':
            activation = tf.nn.sigmoid
        elif name[1] == 'LRELU':
            activation = tf.nn.leaky_relu
        else:
            raise('error: unknown activation function')
        d = name[2:]
        d = [int(x) for x in d]
        return models.GraphMLP_CPLX_dX(settings, d, act=activation)

    elif name[0] == 'CNN':
        if name[1] == 'RELU':
            activation = tf.nn.relu
        elif name[1] == 'TANH':
            activation = tf.nn.tanh
        elif name[1] == 'SIGMOID':
            activation = tf.nn.sigmoid
        elif name[1] == 'LRELU':
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
    elif name[0] == 'ATTNSP':
        if name[1] == 'RELU':
            activation = tf.nn.relu
        elif name[1] == 'TANH':
            activation = tf.nn.tanh
        elif name[1] == 'SIGMOID':
            activation = tf.nn.sigmoid
        elif name[1] == 'LRELU':
            activation = tf.nn.leaky_relu
        else:
            raise('error: unknown activation function')
        d_model = int(name[2])
        ff = int(name[3])
        d = name[4:]
        d = [int(x) for x in d]
        return models.GraphATTNSP_dmodel_ff_dX(settings, d_model, ff, d)
    elif name[0] == 'ATTNMP':
        if name[1] == 'RELU':
            activation = tf.nn.relu
        elif name[1] == 'TANH':
            activation = tf.nn.tanh
        elif name[1] == 'SIGMOID':
            activation = tf.nn.sigmoid
        elif name[1] == 'LRELU':
            activation = tf.nn.leaky_relu
        else:
            raise('error: unknown activation function')
        d_model = int(name[3])
        ff = int(name[4])
        alpha = float(name[2])
        print(alpha)
        d = name[5:]
        d = [int(x) for x in d]
        return models.GraphATTNMP_dmodel_ff_dX(settings, d_model, ff, alpha, d)
    elif name[0] == 'ATTNMPA':
        if name[1] == 'RELU':
            activation = tf.nn.relu
        elif name[1] == 'TANH':
            activation = tf.nn.tanh
        elif name[1] == 'SIGMOID':
            activation = tf.nn.sigmoid
        elif name[1] == 'LRELU':
            activation = tf.nn.leaky_relu
        else:
            raise('error: unknown activation function')
        d_model = int(name[2])
        ff = int(name[3])
        d = name[4:]
        d = [int(x) for x in d]
        return models.GraphATTNMPA_dmodel_ff_dX(settings, d_model, ff, d)
    elif name[0] == 'ATTNMPMH':
        if name[1] == 'RELU':
            activation = tf.nn.relu
        elif name[1] == 'TANH':
            activation = tf.nn.tanh
        elif name[1] == 'SIGMOID':
            activation = tf.nn.sigmoid
        elif name[1] == 'LRELU':
            activation = tf.nn.leaky_relu
        else:
            raise('error: unknown activation function')
        d_model = int(name[3])
        ff = int(name[4])
        alpha = float(name[2])
        print(alpha)
        d = name[5:]
        d = [int(x) for x in d]
        return models.GraphATTNMPMH_dmodel_ff_dX(settings, d_model, ff, alpha, d)

    else:
        raise Exception('error unknown model type')
