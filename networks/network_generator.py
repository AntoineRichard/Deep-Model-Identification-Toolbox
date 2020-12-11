import models
import tensorflow as tf
import warnings
from tensorflow.keras import layers as Layers

#TODO add support for dropout in MLP/CNN

def get_activation(name):
    """
    Parses a string to extract the activation function.
    """
    string = name[1].split(':')

    if string[0] == 'RELU':
        pass
    elif string[0] == 'TANH':
        pass
    elif string[0] == 'SIGMOID':
        pass
    elif string[0] == 'LRELU':
        pass
    elif string[0] == 'PRELU':
        pass
    elif string[0] == 'ELU':
        pass
    elif string[0] == 'SELU':
        pass
    elif string[0] == 'SWISH':
        pass
    elif string[0] == 'CSWISH':
        pass
    elif string[0] == 'MISH':
        pass
    else:
        raise ValueError('error: unknown activation function')
    return name[1]

def CNN_Generator(name,settings):
    """
    Parses a string to extract the model parameters. In this case,
    a CNN (Convolutional Neural Network).
    
    Available layers are the following:
      Convolution layers: 'k' followed by a number, followed by 'c',
                          followed by another number: k3c64 is a 
                          convolution of kernel size 3 with a depth
                          of 64. 
      Pooling layers: 'p' followed by a number: p2 is a pooling layer with
                      a kernel size of 2, with stride 2.
      Dense layers: 'd' followed by a number: d256, d48, d12
                    where the number indicates the number of
                    neurons in that layer.
      Dropout layers: 'r', the keeprate is defined when starting
                      the training.
   
    Syntexically, the string must start with the type of model, then
    the type of activation function must follow, then follows as much
    layers as desired. Each 'word' must be separated using underscores.
  
    A standard CNN would be defined as follows:
      CNN_ (the type of network)
      CNN_RELU_ (the activation function to chose among the available one)
      CNN_RELU_k3c64_k3c64_p2_k5c128_d64_r_d32 (an example of a CNN model)

    All the combinations should work.
    """
    activation = get_activation(name)
    d = name[2:]
    layer_type = []
    layer_param = []
    for di in d:
        if di[0] == 'p':
            layer_type.append('pool')
            layer_param.append(int(di[1]))
        elif di[0] == 'k':
            layer_type.append('conv')
            tmp = di[1:].split('c')
            layer_param.append([int(tmp[0]),int(di[1])])
        elif di[0] == 'd':
            layer_type.append('dense')
            layer_param.append(int(di[1:]))
        elif di[0] == 'r':
            layer_type.append('dropout')
            layer_param.append(0)
        else:
            raise ValueError('Error parsing model name: unknown word')
    return models.GraphCNN(settings, layer_type, layer_param, act=activation)

def MLP_Generator(name,settings):
    """
    Parses a string to extract the model parameters. In this case,
    a MLP (Multi Layer Perceptron).

    Available layers are the following:
      Dense layers: 'd' followed by a number: d256, d48, d12
                    where the number indicates the number of
                    neurons in that layer.
      Dropout layers: 'r', the keeprate is defined when starting
                      the training.
    
    Syntexically, the string must start with the type of model, then
    the type of activation function must follow, then follows as much
    layers as desired. Each 'word' must be separated using underscores.
  
    A standard MLP would be defined as follows:
      MLP_ (the type of network)
      MLP_RELU_ (the activation function to chose among the available one)
      MLP_RELU_d256_d64_r_d32 (an example of a MLP model)

    All the combinations should work.
    """
    activation = get_activation(name)
    d = name[2:]
    layer_type = []
    layer_param = []
    for di in d:
        if di[0] == 'd':
            layer_type.append('dense')
            layer_param.append(int(di[1:]))
        elif di[0] == 'r':
            layer_type.append('dropout')
            layer_param.append(0)
        else:
            raise ValueError('Error parsing model name: unknown word')
    return models.GraphMLP(settings, layer_type, layer_param, act=activation)

def MLP_EVD_Generator(name,settings):
    """
    Parses a string to extract the model parameters. In this case,
    a MLP (Multi Layer Perceptron).

    Available layers are the following:
      Dense layers: 'd' followed by a number: d256, d48, d12
                    where the number indicates the number of
                    neurons in that layer.
      Dropout layers: 'r', the keeprate is defined when starting
                      the training.
    
    Syntexically, the string must start with the type of model, then
    the type of activation function must follow, then follows as much
    layers as desired. Each 'word' must be separated using underscores.
  
    A standard MLP would be defined as follows:
      MLP_ (the type of network)
      MLP_RELU_ (the activation function to chose among the available one)
      MLP_RELU_d256_d64_r_d32 (an example of a MLP model)

    All the combinations should work.
    """
    activation = get_activation(name)
    d = name[2:]
    layer_type = []
    layer_param = []
    for di in d:
        if di[0] == 'd':
            layer_type.append('dense')
            layer_param.append(int(di[1:]))
        elif di[0] == 'r':
            layer_type.append('dropout')
            layer_param.append(0)
        else:
            raise ValueError('Error parsing model name: unknown word')
    return models.GraphEvdMLP(settings, layer_type, layer_param, act=activation)

def MLP_PHY_Generator(name,settings):
    """
    Parses a string to extract the model parameters. In this case,
    a MLP (Multi Layer Perceptron).

    Available layers are the following:
      Dense layers: 'd' followed by a number: d256, d48, d12
                    where the number indicates the number of
                    neurons in that layer.
      Dropout layers: 'r', the keeprate is defined when starting
                      the training.
    
    Syntexically, the string must start with the type of model, then
    the type of activation function must follow, then follows as much
    layers as desired. Each 'word' must be separated using underscores.
  
    A standard MLP would be defined as follows:
      MLP_ (the type of network)
      MLP_RELU_ (the activation function to chose among the available one)
      MLP_RELU_d256_d64_r_d32 (an example of a MLP model)

    All the combinations should work.
    """
    activation = get_activation(name)
    d = name[2:]
    layer_type = []
    layer_param = []
    for di in d:
        if di[0] == 'd':
            layer_type.append('dense')
            layer_param.append(int(di[1:]))
        elif di[0] == 'r':
            layer_type.append('dropout')
            layer_param.append(0)
        else:
            raise ValueError('Error parsing model name: unknown word')
    return models.GraphMLP_PHY(settings, layer_type, layer_param, act=activation)

def MLP_EN_Generator(name,settings):
    """
    Parses a string to extract the model parameters. In this case,
    a MLP (Multi Layer Perceptron).

    Available layers are the following:
      Dense layers: 'd' followed by a number: d256, d48, d12
                    where the number indicates the number of
                    neurons in that layer.
      Dropout layers: 'r', the keeprate is defined when starting
                      the training.
    
    Syntexically, the string must start with the type of model, then
    the type of activation function must follow, then follows as much
    layers as desired. Each 'word' must be separated using underscores.
  
    A standard MLP would be defined as follows:
      MLP_ (the type of network)
      MLP_RELU_ (the activation function to chose among the available one)
      MLP_RELU_d256_d64_r_d32 (an example of a MLP model)

    All the combinations should work.
    """
    activation = get_activation(name)
    d = name[2:]
    layer_type = []
    layer_param = []
    for di in d:
        if di[0] == 'd':
            layer_type.append('dense')
            layer_param.append(int(di[1:]))
        elif di[0] == 'r':
            layer_type.append('dropout')
            layer_param.append(0)
        else:
            raise ValueError('Error parsing model name: unknown word')
    return models.GraphMLP_EN(settings, layer_type, layer_param, act=activation)

def MLP_Complex_Generator(name,settings):
    """
    Parses a string to extract the model parameters. In this case,
    a MLP (Multi Layer Perceptron) based on complex number (as
    opposed to real numbers).
    
    Available layers are the following:
      Dense layers: 'd' followed by a number: d256, d48, d12
                    where the number indicates the number of
                    neurons in that layer.
    
    Syntexically, the string must start with the type of model, then
    the type of activation function must follow, then follows as much
    layers as desired. Each 'word' must be separated using underscores.
    """
    activation = get_activation(name)
    d = name[2:]
    layer_type = []
    layer_param = []
    for di in d:
        if di[0] == 'd':
            layer_type.append('dense')
            layer_param.append(int(di[1:]))
        else:
            raise ValueError('Error parsing model name: unknown word')
    return models.GraphMLP_CPLX(settings, layer_type, layer_param, act=activation)

def RNN_Generator(name,settings):
    """
    Parses a string to extract the model parameters. In this case,
    a simple RNN (Recurrent Neural Network).
    
    Available layers are the following:
      Hidden State Size: 'hs' followed by a number: hs32 means a
                         hidden state of size 32, use this parameter
                         only once. If set multipletime then the
                         parser will select the lastest.
      Recurrent layers: 'l' followed by a number: l3 means that 3 
                        recurrent layers are being stacked onto
                        of each other.
      Dense layers: 'd' followed by a number: d256, d48, d12
                    where the number indicates the number of
                    neurons in that layer.
      Dropout layers: 'r', the keeprate is defined when starting
                      the training.
    
    Syntexically, the string must start with the type of model, then
    the type of activation function must follow, then follows as much
    layers as desired. Each 'word' must be separated using underscores.
  
    A standard RNN would be defined as follows:
    RNN_ (the type of network)
    RNN_RELU_ (the activation function)
    RNN_RELU_hs64_ (the hidden state)
    RNN_RELU_hs64_l2_ (the number of recurrent layers)
    RNN_RELU_hs64_l2_d64_r_d32 (an example of a RNN model)
    
    All the combinations should work.
    """
    activation = get_activation(name)
    hidden_state = None
    recurrent_layers = None
    layer_type = []
    layer_param = []
    d = name[2:]
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
            layer_param.append(0)
        else:
            raise ValueError('Error parsing model name: unknown word')
    return models.GraphRNN(settings, hidden_state, recurrent_layers, layer_type, layer_param, act=activation)

def GRU_Generator(name,settings):
    """
    Parses a string to extract the model parameters. In this case,
    a GRU (Gated Recurrent Unit) network.
    
    Available layers are the following:
      Hidden State Size: 'hs' followed by a number: hs32 means a
                         hidden state of size 32, use this parameter
                         only once. If set multipletime then the
                         parser will select the lastest.
      Recurrent layers: 'l' followed by a number: l3 means that 3 
                        recurrent layers are being stacked onto
                        of each other.
      Dense layers: 'd' followed by a number: d256, d48, d12
                    where the number indicates the number of
                    neurons in that layer.
      Dropout layers: 'r', the keeprate is defined when starting
                      the training.
    
    Syntexically, the string must start with the type of model, then
    the type of activation function must follow, then follows as much
    layers as desired. Each 'word' must be separated using underscores.
  
    A standard RNN would be defined as follows:
    GRU_ (the type of network)
    GRU_RELU_ (the activation function)
    GRU_RELU_hs64_ (the hidden state)
    GRU_RELU_hs64_l2_ (the number of recurrent layers)
    GRU_RELU_hs64_l2_d64_r_d32 (an example of a GRU model)
    
    All the combinations should work.
    """
    activation = get_activation(name)
    hidden_state = None
    recurrent_layers = None
    layer_type = []
    layer_param = []
    d = name[2:]
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
            layer_param.append(0)
        else:
            raise ValueError('Error parsing model name: unknown word')
    return models.GraphGRU(settings, hidden_state, recurrent_layers, layer_type, layer_param, act=activation)

def LSTM_Generator(name,settings):
    """
    Parses a string to extract the model parameters. In this case,
    a LSTM (Long Short Term Memory) network.
    
    Available layers are the following:
      Hidden State Size: 'hs' followed by a number: hs32 means a
                         hidden state of size 32, use this parameter
                         only once. If set multipletime then the
                         parser will select the lastest.
      Recurrent layers: 'l' followed by a number: l3 means that 3 
                        recurrent layers are being stacked onto
                        of each other.
      Dense layers: 'd' followed by a number: d256, d48, d12
                    where the number indicates the number of
                    neurons in that layer.
      Dropout layers: 'r', the keeprate is defined when starting
                      the training.
    
    Syntexically, the string must start with the type of model, then
    the type of activation function must follow, then follows as much
    layers as desired. Each 'word' must be separated using underscores.
  
    A standard RNN would be defined as follows:
    LSTM_ (the type of network)
    LSTM_RELU_ (the activation function)
    LSTM_RELU_hs64_ (the hidden state)
    LSTM_RELU_hs64_l2_ (the number of recurrent layers)
    LSTM_RELU_hs64_l2_d64_r_d32 (an example of a LSTM model)
    
    All the combinations should work.
    """
    activation = get_activation(name)
    hidden_state = None
    recurrent_layers = None
    layer_type = []
    layer_param = []
    d = name[2:]
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
            layer_param.append(0)
        else:
            raise ValueError('Error parsing model name: unknown word')
    return models.GraphLSTM(settings, hidden_state, recurrent_layers, layer_type, layer_param, act=activation)

def ATTNSP_Generator(name,settings):
    """
    Parses a string to extract the model parameters. In this case,
    a Seq2Seq Multi-Head Attention based network.
    
    Available layers are the following:
      model depth: 'md' followed by a number: md64 means that the 
                        embedding .
      Dense layers: 'd' followed by a number: d256, d48, d12
                    where the number indicates the number of
                    neurons in that layer.
      Dropout layers: 'r', the keeprate is defined when starting
                      the training.
    
    Syntexically, the string must start with the type of model, then
    the type of activation function must follow, then follows as much
    layers as desired. Each 'word' must be separated using underscores.
  
    A standard RNN would be defined as follows:
    ATTNSP_ (the type of network)
    ATTNSP_RELU_ (the activation function)
    ATTNSP_RELU_md64_ (the depth of the model)
    ATTNSP_RELU_md64_32_r_32 (an example of a ATTSP model)
    
    All the combinations should work.
    """
    activation = get_activation(name)
    model_depth = None
    layer_type = []
    layer_param = []
    d = name[2:]
    for di in d:
        if di[:2] == 'md':
             model_depth = int(di[2:])
        elif di[0] == 'd':
            layer_type.append('dense')
            layer_param.append(int(di[1:]))
        elif di[0] == 'r':
            layer_type.append('dropout')
            layer_param.append(0)
        else:
            raise ValueError('Error parsing model name: unknown word')
    return models.GraphATTNSP(settings, model_depth, layer_type, layer_param, act=activation)

def ATTNMPMH_Generator(name,settings):
    """
    Parses a string to extract the model parameters. In this case,
    a Seq2Seq Multi-Head Attention based network.
    
    Available layers are the following:
      alpha : 'a' followed by a number: a2 means that the loss value
      of each sequence element will be weighted by its position in the
      sequence to the power 2.
      model depth: 'md' followed by a number: md64 means that the 
                        embedding .
      Dense layers: 'd' followed by a number: d256, d48, d12
                    where the number indicates the number of
                    neurons in that layer.
      Dropout layers: 'r', the keeprate is defined when starting
                      the training.
    
    Syntexically, the string must start with the type of model, then
    the type of activation function must follow, then follows as much
    layers as desired. Each 'word' must be separated using underscores.
  
    A standard RNN would be defined as follows:
    ATTNMPMH_ (the type of network)
    ATTNMPMH_RELU_ (the activation function)
    ATTNMPMH_RELU_a2_ (the loss weighting coefficient)
    ATTNMPMH_RELU_a2_md64_ (the depth of the model)
    ATTNMPMH_RELU_a2_md64_d32_r_d32 (an example of a ATTNMPMH model)
    
    All the combinations should work.
    """
    activation = get_activation(name)
    model_depth = None
    alpha = None
    layer_type = []
    layer_param = []
    d = name[2:]
    for di in d:
        if di[:2] == 'md':
             model_depth = int(di[2:])
        elif di[0] == 'a':
            alpha = float(di[1:])
        elif di[0] == 'd':
            layer_type.append('dense')
            layer_param.append(int(di[1:]))
        elif di[0] == 'r':
            layer_type.append('dropout')
            layer_param.append(0)
        else:
            raise ValueError('Error parsing model name: unknown word')
    return models.GraphATTNMPMH(settings, alpha, model_depth, layer_type, layer_param, act=activation)

def ATTNMPMH_PHY_Generator(name,settings):
    """
    Parses a string to extract the model parameters. In this case,
    a Seq2Seq Multi-Head Attention based network.
    
    Available layers are the following:
      alpha : 'a' followed by a number: a2 means that the loss value
      of each sequence element will be weighted by its position in the
      sequence to the power 2.
      model depth: 'md' followed by a number: md64 means that the 
                        embedding .
      Dense layers: 'd' followed by a number: d256, d48, d12
                    where the number indicates the number of
                    neurons in that layer.
      Dropout layers: 'r', the keeprate is defined when starting
                      the training.
    
    Syntexically, the string must start with the type of model, then
    the type of activation function must follow, then follows as much
    layers as desired. Each 'word' must be separated using underscores.
  
    A standard RNN would be defined as follows:
    ATTNMPMH_ (the type of network)
    ATTNMPMH_RELU_ (the activation function)
    ATTNMPMH_RELU_a2_ (the loss weighting coefficient)
    ATTNMPMH_RELU_a2_md64_ (the depth of the model)
    ATTNMPMH_RELU_a2_md64_d32_r_d32 (an example of a ATTNMPMH model)
    
    All the combinations should work.
    """
    activation = get_activation(name)
    model_depth = None
    alpha = None
    layer_type = []
    layer_param = []
    d = name[2:]
    for di in d:
        if di[:2] == 'md':
             model_depth = int(di[2:])
        elif di[0] == 'a':
            alpha = float(di[1:])
        elif di[0] == 'd':
            layer_type.append('dense')
            layer_param.append(int(di[1:]))
        elif di[0] == 'r':
            layer_type.append('dropout')
            layer_param.append(0)
        else:
            raise ValueError('Error parsing model name: unknown word')
    return models.GraphATTNMPMH_PHY(settings, alpha, model_depth, layer_type, layer_param, act=activation)

def ATTNMP_Generator(name,settings):
    """
    Parses a string to extract the model parameters. In this case,
    a Seq2Seq Attention based network.
    
    Available layers are the following:
      alpha : 'a' followed by a number: a2 means that the loss value
      of each sequence element will be weighted by its position in the
      sequence to the power 2.
      model depth: 'md' followed by a number: md64 means that the 
                        embedding .
      Dense layers: 'd' followed by a number: d256, d48, d12
                    where the number indicates the number of
                    neurons in that layer.
      Dropout layers: 'r', the keeprate is defined when starting
                      the training.
    
    Syntexically, the string must start with the type of model, then
    the type of activation function must follow, then follows as much
    layers as desired. Each 'word' must be separated using underscores.
  
    A standard RNN would be defined as follows:
    ATTNMP_ (the type of network)
    ATTNMP_RELU_ (the activation function)
    ATTNMP_RELU_a2_ (the loss weighting coefficient)
    ATTNMP_RELU_a2_md64_ (the depth of the model)
    ATTNMP_RELU_a2_md64_32_r_32 (an example of a ATTNMP model)
    
    All the combinations should work.
    """
    activation = get_activation(name)
    model_depth = None
    alpha = None
    layer_type = []
    layer_param = []
    d = name[2:]
    for di in d:
        if di[:2] == 'md':
             model_depth = int(di[2:])
        elif di[0] == 'a':
            alpha = float(di[1:])
        elif di[0] == 'd':
            layer_type.append('dense')
            layer_param.append(int(di[1:]))
        elif di[0] == 'r':
            layer_type.append('dropout')
            layer_param.append(0)
        else:
            raise ValueError('Error parsing model name: unknown word')
    return models.GraphATTNMP(settings, alpha, model_depth, layer_type, layer_param, act=activation)

def get_graph(settings,name='net'):
    """
    Generatates advanced models based on a simple string. Each
    condition in the if loop is a different model.
    """
    name = settings.model.split('_')
    if name[0] == 'MLP':
        return MLP_Generator(name,settings)
    if name[0] == 'EVDMLP':
        return MLP_EVD_Generator(name,settings)
    if name[0] == 'MLPPHY':
        return MLP_PHY_Generator(name,settings)
    if name[0] == 'MLPEN':
        return MLP_EN_Generator(name,settings)
    if name[0] == 'MLPCPLX':
        return MLP_Complex_Generator(name,settings)

    elif name[0] == 'CNN':
        return CNN_Generator(name,settings)

    elif name[0] == 'RNN':
        return RNN_Generator(name,settings)

    elif name[0] == 'LSTM':
        return LSTM_Generator(name,settings)

    elif name[0] == 'GRU':
        return GRU_Generator(name,settings)

    elif name[0] == 'ATTNSP':
        return ATTNSP_Generator(name, settings)

    elif name[0] == 'ATTNMP':
        return ATTNMP_Generator(name,settings)

    elif name[0] == 'ATTNMPMH':
        return ATTNMPMH_Generator(name,settings)
    elif name[0] == 'ATTNMPMHPHY':
        return ATTNMPMH_PHY_Generator(name,settings)
    else:
        raise Exception('error unknown model type')
