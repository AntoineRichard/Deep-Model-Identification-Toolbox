import tf_graphv2
import tensorflow as tf

def get_graph(settings):
    if(settings.model[:1] == 'l'):
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
