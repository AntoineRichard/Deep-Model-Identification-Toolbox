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
            activation = tf.nn.leaky_relu
        else:
            raise('error: unknown activation function')
        d = name[2:]
        d = [int(x) for x in d]
        return tf_graphv2.GraphMLP_lX_X(settings, d, act=activation)

    elif name[0] == 'CNN':
        name[2:]
    else:
        raise('error unknown model type')
    if(settings.model[:1] == 'l'`):
        d = settings.model.split('_')[1:]
        d = [int(i) for i in d]
        return tf_graphv2.GraphMLP_lX_X(settings, d)
    elif(network_name.split('_')[1] == 'l1'):
        act_type = network_name.split('_')[0]
        if act_type == 'tanh':
            activation = tf.nn.tanh
        elif act_type == 'leakyrelu':
            activation = tf.nn.leaky_relu
        elif act_type == 'sigmoid':
            activation = tf.nn.sigmoid
        else:
            raise("Unknown activation function found...")
        d = network_name.split('_')[2:]
        d = [int(i) for i in d]
        return tf_graphv2.GraphMLP_l1_X(input_history, input_dim, output_forecast, output_dim, d, act=activation)
    elif(network_name.split('_')[1] == 'l2'):
        act_type = network_name.split('_')[0]
        if act_type == 'tanh':
            activation = tf.nn.tanh
        elif act_type == 'leakyrelu':
            activation = tf.nn.leaky_relu
        elif act_type == 'sigmoid':
            activation = tf.nn.sigmoid
        else:
            raise("Unknown activation function found...")
        d = network_name.split('_')[2:]
        d = [int(i) for i in d]
        return tf_graphv2.GraphMLP_l2_X(input_history, input_dim, output_forecast, output_dim, d, act=activation)
    elif(network_name.split('_')[1] == 'l3'):
        act_type = network_name.split('_')[0]
        if act_type == 'tanh':
            activation = tf.nn.tanh
        elif act_type == 'leakyrelu':
            activation = tf.nn.leaky_relu
        elif act_type == 'sigmoid':
            activation = tf.nn.sigmoid
        else:
            raise("Unknown activation function found...")
        d = network_name.split('_')[2:]
        d = [int(i) for i in d]
        return tf_graphv2.GraphMLP_l3_X(input_history, input_dim, output_forecast, output_dim, d, act=activation)
    elif(network_name.split('_')[1] == 'l4'):
        act_type = network_name.split('_')[0]
        if act_type == 'tanh':
            activation = tf.nn.tanh
        elif act_type == 'leakyrelu':
            activation = tf.nn.leaky_relu
        elif act_type == 'sigmoid':
            activation = tf.nn.sigmoid
        else:
            raise("Unknown activation function found...")
        d = network_name.split('_')[2:]
        d = [int(i) for i in d]
        return tf_graphv2.GraphMLP_l4_X(input_history, input_dim, output_forecast, output_dim, d, act=activation)
