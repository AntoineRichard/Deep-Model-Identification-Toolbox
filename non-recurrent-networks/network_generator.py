import models
import tensorflow as tf

def get_activation(name):
    """
    Parses a string to extract the activation function.
    """
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
    return activation

def CNN_Generator(name):
    """
    Parses a string to extract the model parameters. In this case,
    a CNN (Convolutional Neural Network).
    """
    activation = get_activation(name[1])
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

def MLP_Generator(name):
    """
    Parses a string to extract the model parameters. In this case,
    a MLP (Multi Layer Perceptron).
    """
    activation = get_activation(name[1])
    d = name[2:]
    d = [int(x) for x in d]
    return models.GraphMLP_dX(settings, d, act=activation)

def MLP_Complex_Generator(name):
    """
    Parses a string to extract the model parameters. In this case,
    a MLP (Multi Layer Perceptron) based on complex number (as
    opposed to real numbers).
    """
    activation = get_activation(name[1])
    d = name[2:]
    d = [int(x) for x in d]
    return models.GraphMLP_CPLX_dX(settings, d, act=activation)

def RNN_Generator(name):
    """
    Parses a string to extract the model parameters. In this case,
    a simple RNN (Recurrent Neural Network).
    """
    activation = get_activation(name[1])
    hidden_state = None
    recurrent_layers = None
    layer_type = []
    layer_param = []
    for di in d:
        if di[:2] == 'hs':
            hidden_state = int(di[2:])
        elif di[0] == 'l':
            recurrent_layers = int(di[1:])
        elif di[0] == 'd':
            layer_type.append('dense')
            layer_param.append(int(di[1:]))
        elif di[0] == 'r':
            layer_type.append('dropout')
            layer_param.append(int(di[1:]))
    return models.GraphRNN(settings, hidden_state, recurrent_layers, layer_type, layer_param, act=activation)

def GRU_Generator(name):
    """
    Parses a string to extract the model parameters. In this case,
    a GRU (Gated Recurrent Unit) network.
    """
    activation = get_activation(name[1])
    hidden_state = None
    recurrent_layers = None
    layer_type = []
    layer_param = []
    for di in d:
        if di[:2] == 'hs':
            hidden_state = int(di[2:])
        elif di[0] == 'l':
            recurrent_layers = int(di[1:])
        elif di[0] == 'd':
            layer_type.append('dense')
            layer_param.append(int(di[1:]))
        elif di[0] == 'r':
            layer_type.append('dropout')
            layer_param.append(int(di[1:]))
    return models.GraphGRU(settings, hidden_state, recurrent_layers, layer_type, layer_param, act=activation)

def LSTM_Generator(name):
    """
    Parses a string to extract the model parameters. In this case,
    a LSTM (Long Short Term Memory) network.
    """
    activation = get_activation(name[1])
    hidden_state = None
    recurrent_layers = None
    layer_type = []
    layer_param = []
    for di in d:
        if di[:2] == 'hs':
            hidden_state = int(di[2:])
        elif di[0] == 'l':
            recurrent_layers = int(di[1:])
        elif di[0] == 'd':
            layer_type.append('dense')
            layer_param.append(int(di[1:]))
        elif di[0] == 'r':
            layer_type.append('dropout')
            layer_param.append(int(di[1:]))
    return models.GraphLSTM(settings, hidden_state, recurrent_layers, layer_type, layer_param, act=activation)

def get_graph(settings):
    """
    Generatates advanced models based on a simple string. Each
    condition in the if loop is a different model.
    """
    name = settings.model.split('_')
    if name[0] == 'MLP':
        return MLP_Generator(name)
    
    if name[0] == 'MLPCPLX':
        return MLP_Complex_Generator(name)

    elif name[0] == 'CNN':
        return CNN_Generator(name)

    elif name[0] == 'RNN':
        return RNN_Generator(name)

    elif name[0] == 'LSTM':
        return LSTM_Generator(name)

    elif name[0] == 'GRU':
        return GRU_Generator(name)

    elif name[0] == 'ATTNSP':
        activation = get_activation(name[1])
        d_model = int(name[2])
        ff = int(name[3])
        d = name[4:]
        d = [int(x) for x in d]
        return models.GraphATTNSP_dmodel_ff_dX(settings, d_model, ff, d, act=activation)
    elif name[0] == 'ATTNMP':
        activation = get_activation(name[1])
        d_model = int(name[3])
        ff = int(name[4])
        alpha = float(name[2])
        print(alpha)
        d = name[5:]
        d = [int(x) for x in d]
        return models.GraphATTNMP_dmodel_ff_dX(settings, d_model, ff, alpha, d, act=activation)
    elif name[0] == 'ATTNMPA':
        activation = get_activation(name[1])
        d_model = int(name[2])
        ff = int(name[3])
        d = name[4:]
        d = [int(x) for x in d]
        return models.GraphATTNMPA_dmodel_ff_dX(settings, d_model, ff, d, act=activation)
    elif name[0] == 'ATTNMPMH':
        activation = get_activation(name[1])
        d_model = int(name[3])
        ff = int(name[4])
        alpha = float(name[2])
        print(alpha)
        d = name[5:]
        d = [int(x) for x in d]
        return models.GraphATTNMPMH_dmodel_ff_dX(settings, d_model, ff, alpha, d, act=activation)

    else:
        raise Exception('error unknown model type')
