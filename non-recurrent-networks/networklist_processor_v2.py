import tf_graphv2
import tensorflow as tf

def get_graph(network_name, input_history, input_dim, output_dim):
    if ('k3c2ik3c2id128d128' == network_name):
        return tf_graphv2.GraphCNN_k3c2ik3c2id128d128(input_history,input_dim,output_forecast,output_dim)
    elif('k3c2ik3c2id128d32' == network_name):
        return tf_graphv2.GraphCNN_k3c2ik3c2id128d32(input_history,input_dim,output_forecast,output_dim)
    elif('k3c2ik3c2id128d64' == network_name):
        return tf_graphv2.GraphCNN_k3c2ik3c2id128d64(input_history,input_dim,output_forecast,output_dim)
    elif('k3c2ik3c2id64d32' == network_name):
        return tf_graphv2.GraphCNN_k3c2ik3c2id64d32(input_history,input_dim,output_forecast,output_dim)
    elif('k3c2ik3c2id64d64' == network_name):
        return tf_graphv2.GraphCNN_k3c2ik3c2id64d64(input_history,input_dim,output_forecast,output_dim)
    elif('k3c2ik3c2ik3c2ik3c2id64d64' == network_name):
        return tf_graphv2.GraphCNN_k3c2ik3c2ik3c2ik3c2id64d64(input_history,input_dim,output_forecast,output_dim)
    elif('k3c2ik3c2ip2k3c2ik3c2id64d64' == network_name):
        return tf_graphv2.GraphCNN_k3c2ik3c2ip2k3c2ik3c2id64d64(input_history,input_dim,output_forecast,output_dim)
    elif('k3c2ik3c2ip2k3c2ik3c2ip2d64d64' == network_name):
        return tf_graphv2.GraphCNN_k3c2ik3c2ip2k3c2ik3c2ip2d64d64(input_history,input_dim,output_forecast,output_dim)
    elif('k3c2ik3c2ip2k3c4ik3c4id64d32' == network_name):
        return tf_graphv2.GraphCNN_k3c2ik3c2ip2k3c4ik3c4id64d32(input_history,input_dim,output_forecast,output_dim)
    elif('k3c2ik3c2ip2k3c4ik3c4id64d64' == network_name):
        return tf_graphv2.GraphCNN_k3c2ik3c2ip2k3c4ik3c4id64d64(input_history,input_dim,output_forecast,output_dim)
    elif('k3c2ik3c2ip2k4c2ik4c2ip2d64d32' == network_name):
        return tf_graphv2.GraphCNN_k3c2ik3c2ip2k4c2ik4c2ip2d64d32(input_history,input_dim,output_forecast,output_dim)
    elif('k3c2ik3c2ip2k4c2ik4c2ip2d64d64' == network_name):
        return tf_graphv2.GraphCNN_k3c2ik3c2ip2k4c2ik4c2ip2d64d64(input_history,input_dim,output_forecast,output_dim)
    elif('k3c3ik3c3id128d32' == network_name):
        return tf_graphv2.GraphCNN_k3c3ik3c3id128d32(input_history,input_dim,output_forecast,output_dim)
    elif('k3c3ik3c3id128d64' == network_name):
        return tf_graphv2.GraphCNN_k3c3ik3c3id128d64(input_history,input_dim,output_forecast,output_dim)
    elif('k3c3ik3c3id64d64' == network_name):
        return tf_graphv2.GraphCNN_k3c3ik3c3id64d64(input_history,input_dim,output_forecast,output_dim)
    elif('k3c3ik3c3ip2k3c6ik3c6id64d32' == network_name):
        return tf_graphv2.GraphCNN_k3c3ik3c3ip2k3c6ik3c6id64d32(input_history,input_dim,output_forecast,output_dim)
    elif('k3c3ik3c3ip2k3c6ik3c6id64d64' == network_name):
        return tf_graphv2.GraphCNN_k3c3ik3c3ip2k3c6ik3c6id64d64(input_history,input_dim,output_forecast,output_dim)
    elif('k3c3ik3c3ip2k3c6ik3c6ip2d64d32' == network_name):
        return tf_graphv2.GraphCNN_k3c3ik3c3ip2k3c6ik3c6ip2d64d32(input_history,input_dim,output_forecast,output_dim)
    elif('k3c3ik3c3ip2k3c6ik3c6ip2d64d64' == network_name):
        return tf_graphv2.GraphCNN_k3c3ik3c3ip2k3c6ik3c6ip2d64d64(input_history,input_dim,output_forecast,output_dim)
    elif('k3c4ik3c4id128d32' == network_name):
        return tf_graphv2.GraphCNN_k3c4ik3c4id128d32(input_history,input_dim,output_forecast,output_dim)
    elif('k3c4ik3c4id128d64' == network_name):
        return tf_graphv2.GraphCNN_k3c4ik3c4id128d64(input_history,input_dim,output_forecast,output_dim)
    elif('k3c4ik3c4id64d64' == network_name):
        return tf_graphv2.GraphCNN_k3c4ik3c4id64d64(input_history,input_dim,output_forecast,output_dim)
    elif('k3cik3cid128d32' == network_name):
        return tf_graphv2.GraphCNN_k3cik3cid128d32(input_history,input_dim,output_forecast,output_dim)
    elif('k3cik3cid128d64' == network_name):
        return tf_graphv2.GraphCNN_k3cik3cid128d64(input_history,input_dim,output_forecast,output_dim)
    elif('k3cik3cid64d64' == network_name):
        return tf_graphv2.GraphCNN_k3cik3cid64d64(input_history,input_dim,output_forecast,output_dim)
    elif('k3cik3cip2k3c2ik3c2id64d32' == network_name):
        return tf_graphv2.GraphCNN_k3cik3cip2k3c2ik3c2id64d32(input_history,input_dim,output_forecast,output_dim)
    elif('k3cik3cip2k3c2ik3c2id64d64' == network_name):
        return tf_graphv2.GraphCNN_k3cik3cip2k3c2ik3c2id64d64(input_history,input_dim,output_forecast,output_dim)
    elif('k3cik3cip2k3c2ik3c2ip2d64d32' == network_name):
        return tf_graphv2.GraphCNN_k3cik3cip2k3c2ik3c2ip2d64d32(input_history,input_dim,output_forecast,output_dim)
    elif('k3cik3cip2k3c2ik3c2ip2d64d64' == network_name):
        return tf_graphv2.GraphCNN_k3cik3cip2k3c2ik3c2ip2d64d64(input_history,input_dim,output_forecast,output_dim)
    elif('k4c2ik4c2id128d128' == network_name):
        return tf_graphv2.GraphCNN_k4c2ik4c2id128d128(input_history,input_dim,output_forecast,output_dim)
    elif('k4c2ik4c2id64d32' == network_name):
        return tf_graphv2.GraphCNN_k4c2ik4c2id64d32(input_history,input_dim,output_forecast,output_dim)
    elif('k4c2ik4c2id64d64' == network_name):
        return tf_graphv2.GraphCNN_k4c2ik4c2id64d64(input_history,input_dim,output_forecast,output_dim)
    elif('k5c2ik5c2id128d128' == network_name):
        return tf_graphv2.GraphCNN_k5c2ik5c2id128d128(input_history,input_dim,output_forecast,output_dim)
    elif('k5c2ik5c2id64d32' == network_name):
        return tf_graphv2.GraphCNN_k5c2ik5c2id64d32(input_history,input_dim,output_forecast,output_dim)
    elif('k5c2ik5c2id64d64' == network_name):
        return tf_graphv2.GraphCNN_k5c2ik5c2id64d64(input_history,input_dim,output_forecast,output_dim)
    elif('k5c2ip2k5c2id64d64' == network_name):
        return tf_graphv2.GraphCNN_k5c2ip2k5c2id64d64(input_history,input_dim,output_forecast,output_dim)
    elif('k5c2ip2k5c2ip2d64d64' == network_name):
        return tf_graphv2.GraphCNN_k5c2ip2k5c2ip2d64d64(input_history,input_dim,output_forecast,output_dim)
    elif('k6c2ik6c2id128d128' == network_name):
        return tf_graphv2.GraphCNN_k6c2ik6c2id128d128(input_history,input_dim,output_forecast,output_dim)
    elif('k6c2ik6c2id64d32' == network_name):
        return tf_graphv2.GraphCNN_k6c2ik6c2id64d32(input_history,input_dim,output_forecast,output_dim)
    elif('k6c2ik6c2id64d64' == network_name):
        return tf_graphv2.GraphCNN_k6c2ik6c2id64d64(input_history,input_dim,output_forecast,output_dim)
    elif('k3c2ik3c2ip2k3c4ik3c4ip2d64d32' == network_name):
        return tf_graphv2.GraphCNN_k3c2ik3c2ip2k3c4ik3c4ip2d64d32(input_history,input_dim,output_forecast,output_dim)
    elif('k3c2ik3c2ip2k3c4ik3c4ip2d64d64' == network_name):
        return tf_graphv2.GraphCNN_k3c2ik3c2ip2k3c4ik3c4ip2d64d64(input_history,input_dim,output_forecast,output_dim)
    elif('mh_cnn_k3cik3cip2k3cik3cid64d64' == network_name):
        return tf_graphv2.GraphMHCNN_k3cik3cip2k3cik3cid64d64(input_history,input_dim,output_forecast,output_dim)
    elif('mh_cnn_k3c2ik3c2ip2k3c2ik3c2id64d64' == network_name):
        return tf_graphv2.GraphMHCNN_k3c2ik3c2ip2k3c2ik3c2id64d64(input_history,input_dim,output_forecast,output_dim)
    elif('mh_cnn_k3c3ik3c3ip2k3c3ik3c3id64d64' == network_name):
        return tf_graphv2.GraphMHCNN_k3c3ik3c3ip2k3c3ik3c3id64d64(input_history,input_dim,output_forecast,output_dim)
    elif('mh_cnn_k3c4ik3c4ip2k3c4ik3c4id64d64' == network_name):
        return tf_graphv2.GraphMHCNN_k3c4ik3c4ip2k3c4ik3c4id64d64(input_history,input_dim,output_forecast,output_dim)
    elif('mh_cnn_k3cip2k3cid64d64' == network_name):
        return tf_graphv2.GraphMHCNN_k3cip2k3cid64d64(input_history,input_dim,output_forecast,output_dim)
    elif('mh_cnn_k3c2ip2k3c2id64d64' == network_name):
        return tf_graphv2.GraphMHCNN_k3c2ip2k3c2id64d64(input_history,input_dim,output_forecast,output_dim)
    elif('mh_cnn_k3c3ip2k3c3id64d64' == network_name):
        return tf_graphv2.GraphMHCNN_k3c3ip2k3c3id64d64(input_history,input_dim,output_forecast,output_dim)
    elif('mh_cnn_k3c4ip2k3c4id64d64' == network_name):
        return tf_graphv2.GraphMHCNN_k3c4ip2k3c4id64d64(input_history,input_dim,output_forecast,output_dim)
    elif('mh_cnn_k3cik3cid64d64' == network_name):
        return tf_graphv2.GraphMHCNN_k3cik3cid64d64(input_history,input_dim,output_forecast,output_dim)
    elif('mh_cnn_k3c2ik3c2id64d64' == network_name):
        return tf_graphv2.GraphMHCNN_k3c2ik3c2id64d64(input_history,input_dim,output_forecast,output_dim)
    elif('mh_cnn_k3c3ik3c3id64d64' == network_name):
        return tf_graphv2.GraphMHCNN_k3c3ik3c3id64d64(input_history,input_dim,output_forecast,output_dim)
    elif('mh_cnn_k3c4ik3c4id64d64' == network_name):
        return tf_graphv2.GraphMHCNN_k3c4ik3c4id64d64(input_history,input_dim,output_forecast,output_dim)
    elif('mh_cnn_k3cik3cip2k3cik3cid128d32' == network_name):
        return tf_graphv2.GraphMHCNN_k3cik3cip2k3cik3cid128d32(input_history,input_dim,output_forecast,output_dim)
    elif('mh_cnn_k3c2ik3c2ip2k3c2ik3c2id128d32' == network_name):
        return tf_graphv2.GraphMHCNN_k3c2ik3c2ip2k3c2ik3c2id128d32(input_history,input_dim,output_forecast,output_dim)
    elif('mh_cnn_k3c3ik3c3ip2k3c3ik3c3id128d32' == network_name):
        return tf_graphv2.GraphMHCNN_k3c3ik3c3ip2k3c3ik3c3id128d32(input_history,input_dim,output_forecast,output_dim)
    elif('mh_cnn_k3c4ik3c4ip2k3c4ik3c4id128d32' == network_name):
        return tf_graphv2.GraphMHCNN_k3c4ik3c4ip2k3c4ik3c4id128d32(input_history,input_dim,output_forecast,output_dim)
    elif('mh_cnn_k3cip2k3cid128d32' == network_name):
        return tf_graphv2.GraphMHCNN_k3cip2k3cid128d32(input_history,input_dim,output_forecast,output_dim)
    elif('mh_cnn_k3c2ip2k3c2id128d32' == network_name):
        return tf_graphv2.GraphMHCNN_k3c2ip2k3c2id128d32(input_history,input_dim,output_forecast,output_dim)
    elif('mh_cnn_k3c3ip2k3c3id128d32' == network_name):
        return tf_graphv2.GraphMHCNN_k3c3ip2k3c3id128d32(input_history,input_dim,output_forecast,output_dim)
    elif('mh_cnn_k3c4ip2k3c4id128d32' == network_name):
        return tf_graphv2.GraphMHCNN_k3c4ip2k3c4id128d32(input_history,input_dim,output_forecast,output_dim)
    elif('mh_cnn_k3cik3cid128d32' == network_name):
        return tf_graphv2.GraphMHCNN_k3cik3cid128d32(input_history,input_dim,output_forecast,output_dim)
    elif('mh_cnn_k3c2ik3c2id128d32' == network_name):
        return tf_graphv2.GraphMHCNN_k3c2ik3c2id128d32(input_history,input_dim,output_forecast,output_dim)
    elif('mh_cnn_k3c3ik3c3id128d32' == network_name):
        return tf_graphv2.GraphMHCNN_k3c3ik3c3id128d32(input_history,input_dim,output_forecast,output_dim)
    elif('mh_cnn_k3c4ik3c4id128d32' == network_name):
        return tf_graphv2.GraphMHCNN_k3c4ik3c4id128d32(input_history,input_dim,output_forecast,output_dim)
    elif('mh_cnn_k3cik3cip2k3cik3cid128d64' == network_name):
        return tf_graphv2.GraphMHCNN_k3cik3cip2k3cik3cid128d64(input_history,input_dim,output_forecast,output_dim)
    elif('mh_cnn_k3c2ik3c2ip2k3c2ik3c2id128d64' == network_name):
        return tf_graphv2.GraphMHCNN_k3c2ik3c2ip2k3c2ik3c2id128d64(input_history,input_dim,output_forecast,output_dim)
    elif('mh_cnn_k3c3ik3c3ip2k3c3ik3c3id128d64' == network_name):
        return tf_graphv2.GraphMHCNN_k3c3ik3c3ip2k3c3ik3c3id128d64(input_history,input_dim,output_forecast,output_dim)
    elif('mh_cnn_k3c4ik3c4ip2k3c4ik3c4id128d64' == network_name):
        return tf_graphv2.GraphMHCNN_k3c4ik3c4ip2k3c4ik3c4id128d64(input_history,input_dim,output_forecast,output_dim)
    elif('mh_cnn_k3cip2k3cid128d64' == network_name):
        return tf_graphv2.GraphMHCNN_k3cip2k3cid128d64(input_history,input_dim,output_forecast,output_dim)
    elif('mh_cnn_k3c2ip2k3c2id128d64' == network_name):
        return tf_graphv2.GraphMHCNN_k3c2ip2k3c2id128d64(input_history,input_dim,output_forecast,output_dim)
    elif('mh_cnn_k3c3ip2k3c3id128d64' == network_name):
        return tf_graphv2.GraphMHCNN_k3c3ip2k3c3id128d64(input_history,input_dim,output_forecast,output_dim)
    elif('mh_cnn_k3c4ip2k3c4id128d64' == network_name):
        return tf_graphv2.GraphMHCNN_k3c4ip2k3c4id128d64(input_history,input_dim,output_forecast,output_dim)
    elif('mh_cnn_k3cik3cid128d64' == network_name):
        return tf_graphv2.GraphMHCNN_k3cik3cid128d64(input_history,input_dim,output_forecast,output_dim)
    elif('mh_cnn_k3c2ik3c2id128d64' == network_name):
        return tf_graphv2.GraphMHCNN_k3c2ik3c2id128d64(input_history,input_dim,output_forecast,output_dim)
    elif('mh_cnn_k3c3ik3c3id128d64' == network_name):
        return tf_graphv2.GraphMHCNN_k3c3ik3c3id128d64(input_history,input_dim,output_forecast,output_dim)
    elif('mh_cnn_k3c4ik3c4id128d64' == network_name):
        return tf_graphv2.GraphMHCNN_k3c4ik3c4id128d64(input_history,input_dim,output_forecast,output_dim)
    elif('mh_cnn_k3cXk3cXd64d64' == network_name):
        return tf_graphv2.GraphMHCNN_k3cXk3cXd64d64(input_history,input_dim,output_forecast,output_dim)
    elif('mh_cnn_kXc2ikXc2id64d64' == network_name):
        return tf_graphv2.GraphMHCNN_kXc2ikXc2id64d64(input_history,input_dim,output_forecast,output_dim)
    elif('mh_cnn_kXc3ikXc3id64d64' == network_name):
        return tf_graphv2.GraphMHCNN_kXc3ikXc3id64d64(input_history,input_dim,output_forecast,output_dim)
    elif('mh_cnn_kXc4ikXc4id64d64' == network_name):
        return tf_graphv2.GraphMHCNN_kXc4ikXc4id64d64(input_history,input_dim,output_forecast,output_dim)
    elif('mh_cnn_k3cXk3cXd128d64' == network_name):
        return tf_graphv2.GraphMHCNN_k3cXk3cXd128d64(input_history,input_dim,output_forecast,output_dim)
    elif('mh_cnn_kXc2ikXc2id128d64' == network_name):
        return tf_graphv2.GraphMHCNN_kXc2ikXc2id128d64(input_history,input_dim,output_forecast,output_dim)
    elif('mh_cnn_kXc3ikXc3id128d64' == network_name):
        return tf_graphv2.GraphMHCNN_kXc3ikXc3id128d64(input_history,input_dim,output_forecast,output_dim)
    elif('mh_cnn_kXc4ikXc4id128d64' == network_name):
        return tf_graphv2.GraphMHCNN_kXc4ikXc4id128d64(input_history,input_dim,output_forecast,output_dim)
    elif('mh_cnn_k3cXk3cXd128d32' == network_name):
        return tf_graphv2.GraphMHCNN_k3cXk3cXd128d32(input_history,input_dim,output_forecast,output_dim)
    elif('mh_cnn_kXc2ikXc2id128d32' == network_name):
        return tf_graphv2.GraphMHCNN_kXc2ikXc2id128d32(input_history,input_dim,output_forecast,output_dim)
    elif('mh_cnn_kXc3ikXc3id128d32' == network_name):
        return tf_graphv2.GraphMHCNN_kXc3ikXc3id128d32(input_history,input_dim,output_forecast,output_dim)
    elif('mh_cnn_kXc4ikXc4id128d32' == network_name):
        return tf_graphv2.GraphMHCNN_kXc4ikXc4id128d32(input_history,input_dim,output_forecast,output_dim)
    elif(network_name[:1] == 'l'):
        d = network_name.split('_')[1:]
        d = [int(i) for i in d]
        return tf_graphv2.GraphMLP_lX_X(input_history, input_dim, output_dim, d)
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
