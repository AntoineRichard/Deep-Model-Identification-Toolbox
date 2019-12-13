import numpy as np
import tensorflow as tf

def accuracy(est, gt):
    with tf.name_scope('accuracy_op'):
        diff = tf.sqrt(tf.square(tf.subtract(est, gt)))
        accuracy = tf.reduce_mean(tf.cast(diff, tf.float32),axis = 0)
        tf.summary.scalar('accuracy', tf.reduce_mean(accuracy))
    return accuracy

def train_fn(loss, learning_rate):
    with tf.name_scope('train'):
        tf.summary.scalar('learning_rate', learning_rate)
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_step

class GraphATTNSP_dmodel_ff_dX:
    def __init__(self, settings, d_model, ff, d, act=tf.nn.relu):
        # PLACEHOLDERS
        self.x = tf.placeholder(tf.float32, shape=[None, settings.sequence_length,  settings.input_dim], name='inputs')
        self.y = tf.placeholder(tf.float32, shape=[None, settings.forecast, settings.output_dim], name='target')
        self.step = tf.placeholder(tf.int32, name='step')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
        # Reshape
        mask = self.make_causal_mask(settings.sequence_length)
        tf.summary.image('causal_mask', tf.expand_dims(mask,-1))
        self.yr = tf.reshape(self.y, [-1, settings.forecast*settings.output_dim],name='reshape_target')

        # Embedding
        self.embbed = tf.layers.dense(self.x, d_model, use_bias=False, activation=None)
        # Positional Embedding
        self.pcoded = self.positional_encoding(self.embbed, d_model, settings.sequence_length)
        # Attention Projection
        with tf.name_scope('qkv_projection'):
            self.QW = tf.layers.dense(self.pcoded, d_model, activation=act, name='q_projection')
            self.KW = tf.layers.dense(self.pcoded, d_model, activation=act, name='k_projection')
            self.VW = tf.layers.dense(self.pcoded, d_model, activation=act, name='v_projection')
        # Attention mechanism
        self.xr = self.attention(self.QW, self.KW, self.VW, float(d_model), mask)
        # FeedForward
        with tf.name_scope('feed_forward'):
            for i, di in enumerate(d):
                self.xr = tf.layers.dense(self.xr, di, activation=act, name='dense_'+str(i))
        self.ys_ = tf.layers.dense(self.xr, settings.output_dim, activation=None, name='output_full_dim')
        self.y_ = self.ys_[:,-1,:]#tf.layers.flatten(self.xr, name='output')
        # Loss
        with tf.name_scope('loss_ops'):
            self.diff = tf.square(tf.subtract(self.y_, self.yr))#self.x[:,:,:3]))
            self.s_loss = tf.reduce_mean(self.diff, axis = 1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            tf.summary.scalar('w_loss', self.w_loss)
            tf.summary.scalar('loss', self.s_loss)
        # Train
        self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
        self.acc_op = accuracy(self.y_, self.yr)
        self.train_step = train_fn(self.w_loss, settings.learning_rate)
        # Tensorboard
        self.merged = tf.summary.merge_all()

    def make_causal_mask(self, size):
        with tf.name_scope('causal_mask'):
            return tf.cast(np.tril(np.ones((1,size,size)),k=0)+np.triu(np.ones((1,size,size)),k=1)*1e-9,dtype=tf.float32)


    def attention(self, query, key, value, d_k, mask, name='attention'):
        with tf.variable_scope(name):
            with tf.name_scope('attention_weights'):
                scores = tf.divide(tf.matmul(query, tf.transpose(key, perm=[0, 2, 1])),tf.math.sqrt(d_k))
                masked_scores = tf.multiply(mask, scores)
                self.p_attn = tf.nn.softmax(masked_scores, axis = -1)
                tf.summary.image('attn_weights', tf.expand_dims(self.p_attn,-1))
            #if self.dropout is not None:
            #   self.p_attn = dropout(self.p_attn)
            return tf.matmul(self.p_attn, value)

    def positional_encoding(self, x, d_model, max_len=5000, name='positional_encoding'):
        with tf.variable_scope(name):
            encoding = np.zeros([max_len, d_model], np.float32)
            position = np.arange(max_len, dtype=np.float32)[:, np.newaxis]
            div_term = np.exp(np.arange(0,d_model,2, dtype=np.float32) * (-np.log(10000.0)/d_model))
            encoding[:, 0::2] = np.sin(position*div_term)#, name='sinus_encoding')
            encoding[:, 1::2] = np.cos(position*div_term)#, name='cosinus_encoding')
            return tf.math.add(x, tf.cast(encoding, dtype=tf.float32), name = 'encoding')

class GraphATTNMP_dmodel_ff_dX(GraphATTNSP_dmodel_ff_dX):
    def __init__(self, settings, d_model, ff, alpha, d, act=tf.nn.relu):
        # PLACEHOLDERS
        self.x = tf.placeholder(tf.float32, shape=[None, settings.sequence_length, settings.input_dim], name='inputs')
        self.y = tf.placeholder(tf.float32, shape=[None, settings.sequence_length, settings.output_dim], name='target')
        self.step = tf.placeholder(tf.int32, name='step')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
        # Reshape
        mask = self.make_causal_mask(settings.sequence_length)

        # Embedding
        self.embbed = tf.layers.dense(self.x, d_model, use_bias=False, activation=None)
        # Positional Embedding
        self.pcoded = self.positional_encoding(self.embbed, d_model, settings.sequence_length)
        # Attention Projection
        with tf.name_scope('qkv_projection'):
            self.QW = tf.layers.dense(self.pcoded, d_model, activation=act, name='q_projection')
            self.KW = tf.layers.dense(self.pcoded, d_model, activation=act, name='k_projection')
            self.VW = tf.layers.dense(self.pcoded, d_model, activation=act, name='v_projection')
        # Attention mechanism
        self.xr = self.attention(self.QW, self.KW, self.VW, float(d_model), mask)
        # FeedForward
        with tf.name_scope('feed_forward'):
            for i, di in enumerate(d):
                self.xr = tf.layers.dense(self.xr, di, activation=act, name='dense_'+str(i))
        self.ys_ = tf.layers.dense(self.xr, settings.output_dim, activation=None, name='output_full_dim')
        self.y_ = self.ys_[:,-1,:]#tf.layers.flatten(self.xr, name='output')
        # Loss
        with tf.name_scope('loss_ops'):
            a = tf.range(settings.sequence_length,dtype=tf.float32)
            self.loss_weights = tf.pow(alpha,a)
            tf.summary.histogram('sequence_weights', self.loss_weights)
            self.diff = tf.square(tf.subtract(self.ys_, self.y))
            self.seq_loss = tf.multiply(tf.reduce_mean(self.diff, axis = -1),self.loss_weights)
            self.s_loss = tf.reduce_mean(self.seq_loss, axis = 1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            tf.summary.scalar('w_loss', self.w_loss)
            tf.summary.scalar('loss', tf.reduce_mean(self.s_loss))
        # Train
        self.grad = tf.norm(tf.gradients(self.seq_loss, self.ys_),axis=2)
        self.acc_op = accuracy(self.y_, self.y[:,-1,:])
        self.train_step = train_fn(self.w_loss, settings.learning_rate)
        # Tensorboard
        self.merged = tf.summary.merge_all()

class GraphATTNMPMH_dmodel_ff_dX(GraphATTNSP_dmodel_ff_dX):
    def __init__(self, settings, d_model, ff, alpha, d, act=tf.nn.relu):
        # PLACEHOLDERS
        self.x = tf.placeholder(tf.float32, shape=[None, settings.sequence_length, settings.input_dim], name='inputs')
        self.y = tf.placeholder(tf.float32, shape=[None, settings.sequence_length, settings.output_dim], name='target')
        self.step = tf.placeholder(tf.int32, name='step')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
        # Reshape
        state = self.x[:,:,:settings.input_dim]
        cmd = self.x[:,:,-settings.cmd_dim:]
        mask = self.make_causal_mask(settings.sequence_length)
        tf.summary.image('causal_mask', tf.expand_dims(mask,-1))

        # Embedding
        self.embbed_state = tf.layers.dense(state, d_model, use_bias=False, activation=None, name='state_embedding')
        self.embbed_cmd = tf.layers.dense(cmd, d_model, use_bias=False, activation=None, name='cmd_embedding')
        # Positional Embedding
        self.pcoded_state = self.positional_encoding(self.embbed_state, d_model, settings.sequence_length, name='state_positional_embedding')
        self.pcoded_cmd = self.positional_encoding(self.embbed_cmd, d_model, settings.sequence_length, name='cmd_positional_embedding')
        # Attention Projection
        with tf.name_scope('state_qkv_projection'):
            self.SQW = tf.layers.dense(self.pcoded_state, d_model, activation=act, name='state_q_projection')
            self.SKW = tf.layers.dense(self.pcoded_state, d_model, activation=act, name='state_k_projection')
            self.SVW = tf.layers.dense(self.pcoded_state, d_model, activation=act, name='state_v_projection')
        with tf.name_scope('cmd_qkv_projection'):
            self.CQW = tf.layers.dense(self.pcoded_cmd, d_model, activation=act, name='cmd_q_projection')
            self.CKW = tf.layers.dense(self.pcoded_cmd, d_model, activation=act, name='cmd_k_projection')
            self.CVW = tf.layers.dense(self.pcoded_cmd, d_model, activation=act, name='cmd_v_projection')
        # Attention mechanism
        self.state_attn = self.attention(self.SQW, self.SKW, self.SVW, float(d_model), mask, name='state_attn')
        self.cmd_attn = self.attention(self.CQW, self.CKW, self.CVW, float(d_model), mask, name='cmd_attn')
        self.concat = tf.concat([self.state_attn, self.cmd_attn], axis=-1)
        self.pcoded_fusion = self.positional_encoding(self.concat, d_model*2, settings.sequence_length, name='fusion_positional_embedding')
        with tf.name_scope('fuse_qkv_projection'):
            self.FQW = tf.layers.dense(self.pcoded_fusion, d_model, activation=act, name='fuse_q_projection')
            self.FKW = tf.layers.dense(self.pcoded_fusion, d_model, activation=act, name='fuse_k_projection')
            self.FVW = tf.layers.dense(self.pcoded_fusion, d_model, activation=act, name='fuse_v_projection')
        self.xr = self.attention(self.FQW, self.FKW, self.FVW, float(d_model), mask, name='fused_attn')
        #self.xr = tf.add(self.state_attn, self.cmd_attn)
        #print(self.xr.shape)

        # FeedForward
        with tf.name_scope('feed_forward'):
            for i, di in enumerate(d):
                self.xr = tf.layers.dense(self.xr, di, activation=act, name='dense_'+str(i))
        self.ys_ = tf.layers.dense(self.xr, settings.output_dim, activation=None, name='output_full_dim')
        self.y_ = self.ys_[:,-1,:]#tf.layers.flatten(self.xr, name='output')
        # Loss
        with tf.name_scope('loss_ops'):
            a = tf.range(settings.sequence_length,dtype=tf.float32)
            self.loss_weights = tf.pow(alpha,a)
            tf.summary.histogram('sequence_weights', self.loss_weights)
            self.diff = tf.square(tf.subtract(self.ys_, self.y))
            self.seq_loss = tf.multiply(tf.reduce_mean(self.diff, axis = -1),self.loss_weights)
            self.s_loss = tf.reduce_mean(self.seq_loss, axis = 1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            tf.summary.scalar('w_loss', self.w_loss)
            tf.summary.scalar('loss', tf.reduce_mean(self.s_loss))
        # Train
        self.grad = tf.norm(tf.gradients(self.seq_loss, self.ys_),axis=2)
        self.acc_op = accuracy(self.y_, self.y[:,-1,:])
        self.train_step = train_fn(self.w_loss, settings.learning_rate)
        # Tensorboard
        self.merged = tf.summary.merge_all()

class GraphATTNMPA_dmodel_ff_dX(GraphATTNSP_dmodel_ff_dX):
    def __init__(self, settings, d_model, ff, d, act=tf.nn.relu):
        # PLACEHOLDERS
        self.x = tf.placeholder(tf.float32, shape=[None, settings.sequence_length,settings.input_dim], name='inputs')
        self.y = tf.placeholder(tf.float32, shape=[None, settings.sequence_length, settings.output_dim], name='target')
        self.step = tf.placeholder(tf.int32, name='step')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
        # Reshape
        mask = self.make_causal_mask(settings.sequence_length)

        # Embedding
        self.embbed = tf.layers.dense(self.x, d_model, use_bias=False, activation=None)
        # Positional Embedding
        self.pcoded = self.positional_encoding(self.embbed, d_model, settings.sequence_length)
        # Attention Projection
        with tf.name_scope('qkv_projection'):
            self.QW = tf.layers.dense(self.pcoded, d_model, activation=act, name='q_projection')
            self.KW = tf.layers.dense(self.pcoded, d_model, activation=act, name='k_projection')
            self.VW = tf.layers.dense(self.pcoded, d_model, activation=act, name='v_projection')
        # Attention mechanism
        self.xr = self.attention(self.QW, self.KW, self.VW, float(d_model), mask)
        # FeedForward
        with tf.name_scope('feed_forward'):
            for i, di in enumerate(d):
                self.xr = tf.layers.dense(self.xr, di, activation=act, name='dense_'+str(i))
        self.reg = tf.add(self.xr, self.pcoded)
        self.ys_ = tf.layers.dense(self.reg, settings.output_dim, activation=None, name='output_full_dim')
        self.y_ = self.ys_[:,-1,:]#tf.layers.flatten(self.xr, name='output')
        # Loss
        with tf.name_scope('loss_ops'):
            self.diff = tf.square(tf.subtract(self.ys_, self.y))
            self.s_loss = tf.reduce_mean(tf.reduce_mean(self.diff, axis = 1),axis = 1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            tf.summary.scalar('w_loss', self.w_loss)
            tf.summary.scalar('loss', self.s_loss)
        # Train
        self.grad = tf.norm(tf.gradients(self.seq_loss, self.ys_),axis=2)
        self.acc_op = accuracy(self.y_, self.y[:,-1,:])
        self.train_step = train_fn(self.w_loss, settings.learning_rate)
        # Tensorboard
        self.merged = tf.summary.merge_all()

class GraphATTNMPAR_dmodel_ff_dX(GraphATTNSP_dmodel_ff_dX):
    def __init__(self, settings, d_model, ff, d, act=tf.nn.relu):
        # PLACEHOLDERS
        self.x = tf.placeholder(tf.float32, shape=[None, settings.sequence_length,settings.input_dim], name='inputs')
        self.y = tf.placeholder(tf.float32, shape=[None, settings.sequence_length, settings.output_dim], name='target')
        self.step = tf.placeholder(tf.int32, name='step')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
        # Reshape
        mask = self.make_causal_mask(settings.sequence_length)

        # Embedding
        self.embbed = tf.layers.dense(self.x, d_model, use_bias=False, activation=None)
        # Positional Embedding
        self.pcoded = self.positional_encoding(self.embbed, d_model, settings.sequence_length)
        # Attention Projection
        with tf.name_scope('qkv_projection'):
            self.QW = tf.layers.dense(self.pcoded, d_model, activation=act, name='q_projection')
            self.KW = tf.layers.dense(self.pcoded, d_model, activation=act, name='k_projection')
            self.VW = tf.layers.dense(self.pcoded, d_model, activation=act, name='v_projection')
        # Attention mechanism
        self.xr = self.attention(self.QW, self.KW, self.VW, float(d_model), mask)
        # FeedForward
        with tf.name_scope('feed_forward'):
            for i, di in enumerate(d):
                self.xr = tf.layers.dense(self.xr, di, activation=act, name='dense_'+str(i))
        self.reg = tf.add(self.xr, self.pcoded)
        self.ys_ = tf.layers.dense(self.reg, settings.output_dim, activation=None, name='output_full_dim')
        self.y_ = self.ys_[:,-1,:]#tf.layers.flatten(self.xr, name='output')
        # Loss
        with tf.name_scope('loss_ops'):
            self.diff = tf.square(tf.subtract(self.ys_, self.y))
            self.s_loss = tf.reduce_mean(tf.reduce_mean(self.diff, axis = 1),axis = 1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            tf.summary.scalar('w_loss', self.w_loss)
            tf.summary.scalar('loss', self.s_loss)
        # Train
        self.grad = tf.norm(tf.gradients(self.seq_loss, self.ys_),axis=2)
        self.acc_op = accuracy(self.y_, self.y[:,-1,:])
        self.train_step = train_fn(self.w_loss, settings.learning_rate)
        # Tensorboard
        self.merged = tf.summary.merge_all()

class GraphMLP:
    """
    MLP.
    """
    def __init__(self, settings, layers, params, act=tf.nn.relu):

        # PLACEHOLDERS
        self.x = tf.placeholder(tf.float32, shape=[None, settings.sequence_length,  settings.input_dim], name='inputs')
        self.y = tf.placeholder(tf.float32, shape=[None, settings.forecast, settings.output_dim], name='target')
        self.step = tf.placeholder(tf.int32, name='step')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')

        # Reshape
        self.xr = tf.reshape(self.x, [-1, settings.sequence_length*settings.input_dim],name='reshape_input')
        self.yr = tf.reshape(self.y, [-1, settings.forecast*settings.output_dim],name='reshape_target')
        # Operations
        self.xc = self.xr
        for i, layer_type in enumerate(layers):
            if layer_type == 'dense':
                self.xc = tf.layers.dense(self.xc, params[i], activation=act, name='dense_'+str(i))
            if layer_type == 'dropout':
                self.xc = tf.layers.dropout(self.xc, 1-settings.dropout, name='drop_'+str(i))
        self.y_ = tf.layers.dense(self.xc, settings.output_dim, activation=None, name='output')

        # Loss
        with tf.name_scope('loss_ops'):
            self.diff = tf.square(tf.subtract(self.y_, self.yr))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            tf.summary.scalar('w_loss', self.w_loss)
            tf.summary.scalar('loss', tf.reduce_mean(self.s_loss))
        # Train
        self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
        self.acc_op = accuracy(self.y_, self.yr)
        self.train_step = train_fn(self.w_loss, settings.learning_rate)
        # Tensorboard
        self.merged = tf.summary.merge_all()

class GraphMLP_CPLX:
    """
    MLP for Complex valued number.
    """
    def __init__(self, settings, layer_type, params, act=tf.nn.relu):

        # PLACEHOLDERS
        self.x = tf.placeholder(tf.float32, shape=[None, settings.sequence_length,  settings.input_dim], name='inputs')
        self.y = tf.placeholder(tf.float32, shape=[None, settings.forecast, settings.output_dim], name='target')
        self.step = tf.placeholder(tf.int32, name='step')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')

        # Embed
        self.xe = tf.complex(self.x,self.x)
        # Reshape
        self.xr = tf.reshape(self.xe, [-1, settings.sequence_length*settings.input_dim],name='reshape_input')
        self.yr = tf.reshape(self.y, [-1, settings.forecast*settings.output_dim],name='reshape_target')
        # Operations
        for i, di in enumerate(layer_type):
            if di == 'dense':
                with tf.name_scope('dense_'+str(i)):
                    with tf.name_scope('Complex_Kernel'):
                        wr = tf.Variable(tf.glorot_uniform_initializer()((int(self.xr.shape[-1]), params[i])))
                        wi = tf.Variable(tf.glorot_uniform_initializer()((int(self.xr.shape[-1]), params[i])))
                        weight_matrix = tf.complex(wr,wi)
                    with tf.name_scope('Complex_Bias'):
                        br = tf.Variable(tf.zeros_initializer()(params[i]))
                        bi = tf.Variable(tf.zeros_initializer()(params[i]))
                        bias = tf.complex(br,bi)
                    with tf.name_scope('MatMul'):
                        matmul = tf.matmul(self.xr,weight_matrix)
                    with tf.name_scope('BiasAdd'):
                        biasadd = matmul + bias
                    with tf.name_scope('Activation'):
                        self.xr = tf.tanh(biasadd)
        with tf.name_scope('Complex_Kernel'):
            wr = tf.Variable(tf.glorot_uniform_initializer()((int(self.xr.shape[-1]), settings.output_dim)))
            wi = tf.Variable(tf.glorot_uniform_initializer()((int(self.xr.shape[-1]), settings.output_dim)))
            weight_matrix = tf.complex(wr,wi)
        with tf.name_scope('Complex_Bias'):
            br = tf.Variable(tf.zeros_initializer()(settings.output_dim))
            bi = tf.Variable(tf.zeros_initializer()(settings.output_dim))
            bias = tf.complex(br,bi)
        with tf.name_scope('MatMul'):
            matmul = tf.matmul(self.xr,weight_matrix)
        with tf.name_scope('BiasAdd'):
            biasadd = matmul + bias
        with tf.name_scope('Activation'):
            self.xr = tf.math.real(biasadd)

        self.y_ = tf.layers.dense(self.xr, settings.output_dim,  activation=None, name='output')
        # Loss
        with tf.name_scope('loss_ops'):
            self.diff = tf.square(tf.subtract(self.y_, self.yr))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            tf.summary.scalar('w_loss', self.w_loss)
            tf.summary.scalar('loss', self.s_loss)
        # Train
        self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
        self.acc_op = accuracy(self.y_, self.yr)
        self.train_step = train_fn(self.w_loss, settings.learning_rate)
        # Tensorboard
        self.merged = tf.summary.merge_all()

class GraphCNN:
    """
    CNN.
    """
    def __init__(self, settings, layers, params, act=tf.nn.relu):

        # PLACEHOLDERS
        self.x = tf.placeholder(tf.float32, shape=[None, settings.sequence_length, settings.input_dim], name='inputs')
        self.y = tf.placeholder(tf.float32, shape=[None, settings.forecast, settings.output_dim], name='target')
        self.step = tf.placeholder(tf.int32, name='step')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')

        # Reshape
        self.yr = tf.reshape(self.y, [-1, settings.forecast*settings.output_dim],name='reshape_target')
        # Operations
        must_reshape = True
        self.xc = self.x
        for i, layer_type in enumerate(layers):
            if layer_type == 'conv':
                self.xc = tf.layers.conv1d(self.xc, params[i][0], params[i][1], padding='same', activation=act, name='conv1D_'+str(i))
            if layer_type == 'pool':
                self.xc = tf.layers.max_pooling1d(self.xc, params[i], params[i], padding='same', name='max_pool1D_'+str(i))
            if layer_type == 'dense':
                if must_reshape:
                    self.xc = tf.layers.flatten(self.xc, name='flatten')
                    must_reshape = False
                self.xc = tf.layers.dense(self.xc, params[i], activation=act, name='dense_'+str(i))
            if layer_type == 'dropout':
                self.xc = tf.layers.dropout(self.xc, 1-settings.dropout, name='drop_'+str(i))
        self.y_ = tf.layers.dense(self.xc, settings.output_dim, activation=None, name='output')

        # Loss
        with tf.name_scope('loss_ops'):
            self.diff = tf.square(tf.subtract(self.y_, self.yr))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            tf.summary.scalar('w_loss', self.w_loss)
            tf.summary.scalar('loss', tf.reduce_mean(self.s_loss))
        # Train
        self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
        self.acc_op = accuracy(self.y_, self.yr)
        self.train_step = train_fn(self.w_loss, settings.learning_rate)
        # Tensorboard
        self.merged = tf.summary.merge_all()

class GraphRNN:
    """
    RNN.
    """
    def __init__(self, settings, hidden_state, recurrent_layers, layer_type, params, act=tf.nn.relu):
        self.v_recurrent_layers = recurrent_layers
        self.v_hidden_state = hidden_state
        # PLACEHOLDERS
        self.x = tf.placeholder(tf.float32, [None, settings.sequence_length, settings.input_dim], name = 'input')
        self.y = tf.placeholder(tf.float32, [None, settings.sequence_length, settings.output_dim], name = 'target')
        self.hs = tf.placeholder(tf.float32, [recurrent_layers, None, hidden_state], name = 'hidden_state')
        self.step = tf.placeholder(tf.int32, name='step')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
       
        # Hidden-State definition 
        self.rnn_tuple_state = tuple([self.hs[idx] for idx in range(recurrent_layers)])
        stacked_rnn = []
        for _ in range(recurrent_layers):
            stacked_rnn.append(tf.nn.rnn_cell.BasicRNNCell(hidden_state, activation=tf.nn.tanh, name='rnn_cell_'+str(_)))
        cell = tf.nn.rnn_cell.MultiRNNCell(stacked_rnn, state_is_tuple = True)
        # Reshape data
        self.xr = tf.reshape(self.x, [-1, settings.sequence_length*settings.input_dim], name="input-reformated")
        self.yr = tf.reshape(self.y, [-1, settings.sequence_length*settings.output_dim], name="target-reformated")
        self.inputs_series = tf.split(self.xr, settings.sequence_length, axis=1)
        self.labels_series = tf.split(self.yr, settings.sequence_length, axis=1)
        # Send data in two passes
        self.input_one = [self.inputs_series[0]]
        self.input_two = self.inputs_series[1:]
        # Operations
        # Roll RNN in 2 steps
        # First to get the the hidden state after only one iteration (needed for multistep)
        self.first_series, self.mid_state = tf.nn.static_rnn(cell, self.input_one, initial_state=self.rnn_tuple_state)
        # Then roll the rest
        self.second_series, self.current_state = tf.nn.static_rnn(cell, self.input_two, initial_state=self.mid_state)
        self.states_series = self.first_series + self.second_series
        # FeedForward layers
        self.tensor_state = tf.transpose(tf.convert_to_tensor(self.states_series),[1,0,2])
        self.xc = self.tensor_state
        for i, layer in enumerate(layer_type):
            if layer == 'dense':
                self.xc = tf.layers.dense(self.xc, params[i], activation=act, name='dense_'+str(i))
            if layer == 'dropout':
                self.xc = tf.layers.dropout(self.xc, 1-settings.dropout, name='drop_'+str(i))
        self.y_ = tf.layers.dense(self.xc, settings.output_dim, activation=None, name='output')
        # Losses
        with tf.name_scope('loss_ops'):
            self.diff = tf.square(tf.subtract(self.y_, self.y))
            self.s_loss =  tf.reduce_mean(tf.reduce_mean(self.diff, axis=1),axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            tf.summary.scalar('w_loss', self.w_loss)
            tf.summary.scalar('loss', tf.reduce_mean(self.s_loss))
        # Train
        self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
        self.acc_op = accuracy(self.y_[-1], self.y[-1])
        self.train_step = train_fn(self.w_loss, settings.learning_rate)
        # Tensorboard
        self.merged = tf.summary.merge_all()

    def get_hidden_state(self, batch_size):
        return np.zeros((self.v_recurrent_layers, batch_size, self.v_hidden_state))

class GraphGRU:
    """
    GRU.
    """
    def __init__(self, settings, hidden_state, recurrent_layers, layer_type, params, act=tf.nn.relu):
        self.v_recurrent_layers = recurrent_layers
        self.v_hidden_state = hidden_state
        # PLACEHOLDERS
        self.x = tf.placeholder(tf.float32, [None, settings.sequence_length, settings.input_dim], name = 'input')
        self.y = tf.placeholder(tf.float32, [None, settings.sequence_length, settings.output_dim], name = 'target')
        self.hs = tf.placeholder(tf.float32, [recurrent_layers, None, hidden_state], name = 'hidden_state')
        self.step = tf.placeholder(tf.int32, name='step')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
       
        # Hidden-State definition 
        self.rnn_tuple_state = tuple([self.hs[idx] for idx in range(recurrent_layers)])
        stacked_rnn = []
        for _ in range(recurrent_layers):
            stacked_rnn.append(tf.nn.rnn_cell.GRUCell(hidden_state, activation=tf.nn.tanh, name='gru_cell_'+str(_)))
        cell = tf.nn.rnn_cell.MultiRNNCell(stacked_rnn, state_is_tuple = True)
        # Reshape data
        self.xr = tf.reshape(self.x, [-1, settings.sequence_length*settings.input_dim], name="input-reformated")
        self.yr = tf.reshape(self.y, [-1, settings.sequence_length*settings.output_dim], name="target-reformated")
        self.inputs_series = tf.split(self.xr, settings.sequence_length, axis=1)
        self.labels_series = tf.split(self.yr, settings.sequence_length, axis=1)
        # Send data in two passes
        self.input_one = [self.inputs_series[0]]
        self.input_two = self.inputs_series[1:]
        # Operations
        # Roll RNN in 2 steps
        # First to get the the hidden state after only one iteration (needed for multistep)
        self.first_series, self.mid_state = tf.nn.static_rnn(cell, self.input_one, initial_state=self.rnn_tuple_state)
        # Then roll the rest
        self.second_series, self.current_state = tf.nn.static_rnn(cell, self.input_two, initial_state=self.mid_state)
        self.states_series = self.first_series + self.second_series
        # FeedForward layers
        self.tensor_state = tf.transpose(tf.convert_to_tensor(self.states_series),[1,0,2])
        self.xc = self.tensor_state
        for i, layer in enumerate(layer_type):
            if layer == 'dense':
                self.xc = tf.layers.dense(self.xc, params[i], activation=act, name='dense_'+str(i))
            if layer == 'dropout':
                self.xc = tf.layers.dropout(self.xc, 1-settings.dropout, name='drop_'+str(i))
        self.y_ = tf.layers.dense(self.xc, settings.output_dim, activation=None, name='output')
        # Losses
        with tf.name_scope('loss_ops'):
            self.diff = tf.square(tf.subtract(self.y_, self.y))
            self.s_loss =  tf.reduce_mean(tf.reduce_mean(self.diff, axis=1),axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            tf.summary.scalar('w_loss', self.w_loss)
            tf.summary.scalar('loss', tf.reduce_mean(self.s_loss))
        # Train
        self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
        self.acc_op = accuracy(self.y_[-1], self.y[-1])
        self.train_step = train_fn(self.w_loss, settings.learning_rate)
        # Tensorboard
        self.merged = tf.summary.merge_all()
    
    def get_hidden_state(self, batch_size):
        return np.zeros((self.v_recurrent_layers, batch_size, self.v_hidden_state))



class GraphMHCNN_k3cik3cip2k3cik3cid64d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(c1, input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d2_br_'+str(branch))
                p1 = tf.layers.max_pooling1d(c2, 2, 2, padding='same', name='pool1d1_br_'+str(branch))
                c3 = tf.layers.conv1d(p1, input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d3_br_'+str(branch))
                c4 = tf.layers.conv1d(c3, input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c4, [-1, int(input_dim*input_history/2)],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 64, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3c2ik3c2ip2k3c2ik3c2id64d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(c1, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d2_br_'+str(branch))
                p1 = tf.layers.max_pooling1d(c2, 2, 2, padding='same', name='pool1d1_br_'+str(branch))
                c3 = tf.layers.conv1d(p1, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d3_br_'+str(branch))
                c4 = tf.layers.conv1d(c3, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c4, [-1, input_dim*input_history],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 64, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3c3ik3c3ip2k3c3ik3c3id64d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(c1, input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d2_br_'+str(branch))
                p1 = tf.layers.max_pooling1d(c2, 2, 2, padding='same', name='pool1d1_br_'+str(branch))
                c3 = tf.layers.conv1d(p1, input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d3_br_'+str(branch))
                c4 = tf.layers.conv1d(c3, input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c4, [-1, int(input_dim*input_history*3/2)],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 64, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3c4ik3c4ip2k3c4ik3c4id64d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(c1, input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d2_br_'+str(branch))
                p1 = tf.layers.max_pooling1d(c2, 2, 2, padding='same', name='pool1d1_br_'+str(branch))
                c3 = tf.layers.conv1d(p1, input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d3_br_'+str(branch))
                c4 = tf.layers.conv1d(c3, input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c4, [-1, input_dim*input_history*2],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 64, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3cip2k3cid64d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                p1 = tf.layers.max_pooling1d(c1, 2, 2, padding='same', name='pool1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(p1, input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c2, [-1, int(input_dim*input_history/2)],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 64, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3c2ip2k3c2id64d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():
            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')
            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                p1 = tf.layers.max_pooling1d(c1, 2, 2, padding='same', name='pool1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(p1, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c2, [-1, input_dim*input_history],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 64, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')
            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3c3ip2k3c3id64d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                p1 = tf.layers.max_pooling1d(c1, 2, 2, padding='same', name='pool1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(p1, input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c2, [-1, int(input_dim*input_history*3/2)],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 64, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3c4ip2k3c4id64d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                p1 = tf.layers.max_pooling1d(c1, 2, 2, padding='same', name='pool1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(p1, input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c2, [-1, input_dim*input_history*2],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 64, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3cik3cid64d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(c1, input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c2, [-1, input_dim*input_history],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 64, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3c2ik3c2id64d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(c1, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c2, [-1, input_dim*input_history*2],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 64, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3c3ik3c3id64d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(c1, input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c2, [-1, input_dim*input_history*3],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 64, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3c4ik3c4id64d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(c1, input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c2, [-1, input_dim*input_history*4],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 64, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)

class GraphMHCNN_k3cik3cip2k3cik3cid128d32:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(c1, input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d2_br_'+str(branch))
                p1 = tf.layers.max_pooling1d(c2, 2, 2, padding='same', name='pool1d1_br_'+str(branch))
                c3 = tf.layers.conv1d(p1, input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d3_br_'+str(branch))
                c4 = tf.layers.conv1d(c3, input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c4, [-1, int(input_dim*input_history/2)],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 32, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3c2ik3c2ip2k3c2ik3c2id128d32:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(c1, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d2_br_'+str(branch))
                p1 = tf.layers.max_pooling1d(c2, 2, 2, padding='same', name='pool1d1_br_'+str(branch))
                c3 = tf.layers.conv1d(p1, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d3_br_'+str(branch))
                c4 = tf.layers.conv1d(c3, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c4, [-1, input_dim*input_history],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 32, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3c3ik3c3ip2k3c3ik3c3id128d32:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(c1, input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d2_br_'+str(branch))
                p1 = tf.layers.max_pooling1d(c2, 2, 2, padding='same', name='pool1d1_br_'+str(branch))
                c3 = tf.layers.conv1d(p1, input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d3_br_'+str(branch))
                c4 = tf.layers.conv1d(c3, input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c4, [-1, int(input_dim*input_history*3/2)],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 32, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3c4ik3c4ip2k3c4ik3c4id128d32:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(c1, input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d2_br_'+str(branch))
                p1 = tf.layers.max_pooling1d(c2, 2, 2, padding='same', name='pool1d1_br_'+str(branch))
                c3 = tf.layers.conv1d(p1, input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d3_br_'+str(branch))
                c4 = tf.layers.conv1d(c3, input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c4, [-1, input_dim*input_history*2],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 32, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3cip2k3cid128d32:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                p1 = tf.layers.max_pooling1d(c1, 2, 2, padding='same', name='pool1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(p1, input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c2, [-1, int(input_dim*input_history/2)],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 32, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3c2ip2k3c2id128d32:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():
            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')
            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                p1 = tf.layers.max_pooling1d(c1, 2, 2, padding='same', name='pool1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(p1, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c2, [-1, input_dim*input_history],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 32, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')
            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3c3ip2k3c3id128d32:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                p1 = tf.layers.max_pooling1d(c1, 2, 2, padding='same', name='pool1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(p1, input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c2, [-1, int(input_dim*input_history*3/2)],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 32, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3c4ip2k3c4id128d32:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                p1 = tf.layers.max_pooling1d(c1, 2, 2, padding='same', name='pool1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(p1, input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c2, [-1, input_dim*input_history*2],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 32, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3cik3cid128d32:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(c1, input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c2, [-1, input_dim*input_history],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 32, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3c2ik3c2id128d32:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(c1, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c2, [-1, input_dim*input_history*2],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 32, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3c3ik3c3id128d32:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(c1, input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c2, [-1, input_dim*input_history*3],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 32, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3c4ik3c4id128d32:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(c1, input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c2, [-1, input_dim*input_history*4],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 32, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)

class GraphMHCNN_k3cik3cip2k3cik3cid128d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(c1, input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d2_br_'+str(branch))
                p1 = tf.layers.max_pooling1d(c2, 2, 2, padding='same', name='pool1d1_br_'+str(branch))
                c3 = tf.layers.conv1d(p1, input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d3_br_'+str(branch))
                c4 = tf.layers.conv1d(c3, input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c4, [-1, int(input_dim*input_history/2)],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3c2ik3c2ip2k3c2ik3c2id128d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(c1, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d2_br_'+str(branch))
                p1 = tf.layers.max_pooling1d(c2, 2, 2, padding='same', name='pool1d1_br_'+str(branch))
                c3 = tf.layers.conv1d(p1, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d3_br_'+str(branch))
                c4 = tf.layers.conv1d(c3, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c4, [-1, input_dim*input_history],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3c3ik3c3ip2k3c3ik3c3id128d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(c1, input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d2_br_'+str(branch))
                p1 = tf.layers.max_pooling1d(c2, 2, 2, padding='same', name='pool1d1_br_'+str(branch))
                c3 = tf.layers.conv1d(p1, input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d3_br_'+str(branch))
                c4 = tf.layers.conv1d(c3, input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c4, [-1, int(input_dim*input_history*3/2)],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3c4ik3c4ip2k3c4ik3c4id128d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(c1, input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d2_br_'+str(branch))
                p1 = tf.layers.max_pooling1d(c2, 2, 2, padding='same', name='pool1d1_br_'+str(branch))
                c3 = tf.layers.conv1d(p1, input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d3_br_'+str(branch))
                c4 = tf.layers.conv1d(c3, input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c4, [-1, input_dim*input_history*2],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3cip2k3cid128d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                p1 = tf.layers.max_pooling1d(c1, 2, 2, padding='same', name='pool1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(p1, input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c2, [-1, int(input_dim*input_history/2)],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3c2ip2k3c2id128d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():
            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')
            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                p1 = tf.layers.max_pooling1d(c1, 2, 2, padding='same', name='pool1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(p1, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c2, [-1, input_dim*input_history],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')
            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3c3ip2k3c3id128d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                p1 = tf.layers.max_pooling1d(c1, 2, 2, padding='same', name='pool1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(p1, input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c2, [-1, int(input_dim*input_history*3/2)],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3c4ip2k3c4id128d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                p1 = tf.layers.max_pooling1d(c1, 2, 2, padding='same', name='pool1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(p1, input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c2, [-1, input_dim*input_history*2],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3cik3cid128d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(c1, input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c2, [-1, input_dim*input_history],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3c2ik3c2id128d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(c1, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c2, [-1, input_dim*input_history*2],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3c3ik3c3id128d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(c1, input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c2, [-1, input_dim*input_history*3],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3c4ik3c4id128d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            for branch in range(input_dim):
                c1 = tf.layers.conv1d(self.rs[:,:,branch:branch+1], input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_br_'+str(branch))
                c2 = tf.layers.conv1d(c1, input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_br_'+str(branch))
                rs2 = tf.reshape(c2, [-1, input_dim*input_history*4],name='rs_br_'+str(branch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)

class GraphMHCNN_k3cXk3cXd64d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            channel_list = [input_dim,2*input_dim,3*input_dim]
            for ch in channel_list:
                c1 = tf.layers.conv1d(self.rs, ch, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_ker_'+str(ch))
                c2 = tf.layers.conv1d(c1, ch, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_ker_'+str(ch))
                rs2 = tf.reshape(c2, [-1, ch*input_history],name='rs_ker_'+str(ch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 64, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_kXc2ikXc2id64d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            kernel_list = [3,4,5,6]
            for ker in kernel_list:
                c1 = tf.layers.conv1d(self.rs, input_dim*2, ker, padding = 'same',activation=tf.nn.relu ,name='conv1d1_ker_'+str(ker))
                c2 = tf.layers.conv1d(c1, input_dim*2, ker, padding = 'same',activation=tf.nn.relu ,name='conv1d4_ker_'+str(ker))
                rs2 = tf.reshape(c2, [-1, input_dim*input_history*2],name='rs_ker_'+str(ker))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 64, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_kXc3ikXc3id64d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            kernel_list = [3,4,5,6]
            for ker in kernel_list:
                c1 = tf.layers.conv1d(self.rs, input_dim*3, ker, padding = 'same',activation=tf.nn.relu ,name='conv1d1_ker_'+str(ker))
                c2 = tf.layers.conv1d(c1, input_dim*3, ker, padding = 'same',activation=tf.nn.relu ,name='conv1d4_ker_'+str(ker))
                rs2 = tf.reshape(c2, [-1, input_dim*input_history*3],name='rs_ker_'+str(ker))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 64, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_kXc4ikXc4id64d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            kernel_list = [3,4,5,6]
            for ker in kernel_list:
                c1 = tf.layers.conv1d(self.rs, input_dim*4, ker, padding = 'same',activation=tf.nn.relu ,name='conv1d1_ker_'+str(ker))
                c2 = tf.layers.conv1d(c1, input_dim*4, ker, padding = 'same',activation=tf.nn.relu ,name='conv1d4_ker_'+str(ker))
                rs2 = tf.reshape(c2, [-1, input_dim*input_history*4],name='rs_ker_'+str(ker))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 64, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3cXk3cXd128d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            channel_list = [input_dim,2*input_dim,3*input_dim]
            for ch in channel_list:
                c1 = tf.layers.conv1d(self.rs, ch, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_ker_'+str(ch))
                c2 = tf.layers.conv1d(c1, ch, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_ker_'+str(ch))
                rs2 = tf.reshape(c2, [-1, ch*input_history],name='rs_ker_'+str(ch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_kXc2ikXc2id128d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            kernel_list = [3,4,5,6]
            for ker in kernel_list:
                c1 = tf.layers.conv1d(self.rs, input_dim*2, ker, padding = 'same',activation=tf.nn.relu ,name='conv1d1_ker_'+str(ker))
                c2 = tf.layers.conv1d(c1, input_dim*2, ker, padding = 'same',activation=tf.nn.relu ,name='conv1d4_ker_'+str(ker))
                rs2 = tf.reshape(c2, [-1, input_dim*input_history*2],name='rs_ker_'+str(ker))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_kXc3ikXc3id128d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            kernel_list = [3,4,5,6]
            for ker in kernel_list:
                c1 = tf.layers.conv1d(self.rs, input_dim*3, ker, padding = 'same',activation=tf.nn.relu ,name='conv1d1_ker_'+str(ker))
                c2 = tf.layers.conv1d(c1, input_dim*3, ker, padding = 'same',activation=tf.nn.relu ,name='conv1d4_ker_'+str(ker))
                rs2 = tf.reshape(c2, [-1, input_dim*input_history*3],name='rs_ker_'+str(ker))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_kXc4ikXc4id128d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            kernel_list = [3,4,5,6]
            for ker in kernel_list:
                c1 = tf.layers.conv1d(self.rs, input_dim*4, ker, padding = 'same',activation=tf.nn.relu ,name='conv1d1_ker_'+str(ker))
                c2 = tf.layers.conv1d(c1, input_dim*4, ker, padding = 'same',activation=tf.nn.relu ,name='conv1d4_ker_'+str(ker))
                rs2 = tf.reshape(c2, [-1, input_dim*input_history*4],name='rs_ker_'+str(ker))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_k3cXk3cXd128d32:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            channel_list = [input_dim,2*input_dim,3*input_dim]
            for ch in channel_list:
                c1 = tf.layers.conv1d(self.rs, ch, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1_ker_'+str(ch))
                c2 = tf.layers.conv1d(c1, ch, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4_ker_'+str(ch))
                rs2 = tf.reshape(c2, [-1, ch*input_history],name='rs_ker_'+str(ch))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 32, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_kXc2ikXc2id128d32:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            kernel_list = [3,4,5,6]
            for ker in kernel_list:
                c1 = tf.layers.conv1d(self.rs, input_dim*2, ker, padding = 'same',activation=tf.nn.relu ,name='conv1d1_ker_'+str(ker))
                c2 = tf.layers.conv1d(c1, input_dim*2, ker, padding = 'same',activation=tf.nn.relu ,name='conv1d4_ker_'+str(ker))
                rs2 = tf.reshape(c2, [-1, input_dim*input_history*2],name='rs_ker_'+str(ker))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 32, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)
class GraphMHCNN_kXc3ikXc3id128d32:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            kernel_list = [3,4,5,6]
            for ker in kernel_list:
                c1 = tf.layers.conv1d(self.rs, input_dim*3, ker, padding = 'same',activation=tf.nn.relu ,name='conv1d1_ker_'+str(ker))
                c2 = tf.layers.conv1d(c1, input_dim*3, ker, padding = 'same',activation=tf.nn.relu ,name='conv1d4_ker_'+str(ker))
                rs2 = tf.reshape(c2, [-1, input_dim*input_history*3],name='rs_ker_'+str(ker))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 32, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)

class GraphMHCNN_kXc4ikXc4id128d32:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # Operations
            flats  = []
            kernel_list = [3,4,5,6]
            for ker in kernel_list:
                c1 = tf.layers.conv1d(self.rs, input_dim*4, ker, padding = 'same',activation=tf.nn.relu ,name='conv1d1_ker_'+str(ker))
                c2 = tf.layers.conv1d(c1, input_dim*4, ker, padding = 'same',activation=tf.nn.relu ,name='conv1d4_ker_'+str(ker))
                rs2 = tf.reshape(c2, [-1, input_dim*input_history*4],name='rs_ker_'+str(ker))
                flats.append(rs2)

            self.heads = tf.concat(flats,-1)
            self.d1 = tf.layers.dense(self.heads, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 32, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)

