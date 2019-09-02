
import tensorflow as tf

def accuracy(estimation):
    accuracy = tf.reduce_mean(tf.cast(estimation, tf.float32),axis = 0)
    return accuracy

def train_fn(loss, learning_rate):
    with tf.variable_scope('train'):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_step

class GraphMLP_l1_X:
    def __init__(self,input_history, input_dim, output_forecast, output_dim, d, act=tf.nn.relu):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions

        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')

            # Operations
            self.d1 = tf.layers.dense(self.x, d[0], activation=act, name='dense1')
            self.y_ = tf.layers.dense(self.d1, output_dim, activation=None, name='output')
            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)

class GraphMLP_l2_X:
    def __init__(self,input_history, input_dim, output_forecast, output_dim, d, act=tf.nn.relu):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions

        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')

            # Operations
            self.d1 = tf.layers.dense(self.x, d[0], activation=act, name='dense1')
            self.d2 = tf.layers.dense(self.d1, d[1], activation=act, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')
            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)

class GraphMLP_l3_X:
    def __init__(self,input_history, input_dim, output_forecast, output_dim, d, act=tf.nn.relu):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions

        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')

            # Operations
            self.d1 = tf.layers.dense(self.x, d[0], activation=act, name='dense1')
            self.d2 = tf.layers.dense(self.d1, d[1], activation=act, name='dense2')
            self.d3 = tf.layers.dense(self.d2, d[2], activation=act, name='dense3')
            self.y_ = tf.layers.dense(self.d3, output_dim, activation=None, name='output')
            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)

class GraphMLP_l4_X:
    def __init__(self,input_history, input_dim, output_forecast, output_dim, d, act=tf.nn.relu):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions

        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')

            # Operations
            self.d1 = tf.layers.dense(self.x, d[0], activation=act, name='dense1')
            self.d2 = tf.layers.dense(self.d1, d[1], activation=act, name='dense2')
            self.d3 = tf.layers.dense(self.d2, d[2], activation=act, name='dense3')
            self.d4 = tf.layers.dense(self.d3, d[3], activation=act, name='dense4')
            self.y_ = tf.layers.dense(self.d4, output_dim, activation=None, name='output')
            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)

class GraphCNN_k4c2ik4c2id64d64:
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
            self.c1 = tf.layers.conv1d(self.rs, input_dim*2, 4, padding = 'same',activation=tf.nn.relu ,name='conv1d1')
            self.c2 = tf.layers.conv1d(self.c1, input_dim*2, 4, padding = 'same',activation=tf.nn.relu ,name='conv1d2')
            self.rs2 = tf.reshape(self.c2, [-1, input_dim*2*input_history],name='output')
            self.d1 = tf.layers.dense(self.rs2, 64, activation=tf.nn.relu, name='dense1')
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

class GraphCNN_k5c2ik5c2id64d64:
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
            self.c1 = tf.layers.conv1d(self.rs, input_dim*2, 5, padding = 'same',activation=tf.nn.relu ,name='conv1d1')
            self.c2 = tf.layers.conv1d(self.c1, input_dim*2, 5, padding = 'same',activation=tf.nn.relu ,name='conv1d2')
            self.rs2 = tf.reshape(self.c2, [-1, input_dim*2*input_history],name='output')
            self.d1 = tf.layers.dense(self.rs2, 64, activation=tf.nn.relu, name='dense1')
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

class GraphCNN_k6c2ik6c2id64d64:
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
            self.c1 = tf.layers.conv1d(self.rs, input_dim*2, 4, padding = 'same',activation=tf.nn.relu ,name='conv1d1')
            self.c2 = tf.layers.conv1d(self.c1, input_dim*2, 4, padding = 'same',activation=tf.nn.relu ,name='conv1d2')
            self.rs2 = tf.reshape(self.c2, [-1, input_dim*2*input_history],name='output')
            self.d1 = tf.layers.dense(self.rs2, 64, activation=tf.nn.relu, name='dense1')
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

class GraphCNN_k3c2ik3c2id128d128:
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
            self.c1 = tf.layers.conv1d(self.rs, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1')
            self.c2 = tf.layers.conv1d(self.c1, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d2')
            self.rs2 = tf.reshape(self.c2, [-1, input_dim*2*input_history],name='output')
            self.d1 = tf.layers.dense(self.rs2, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 128, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)

class GraphCNN_k4c2ik4c2id128d128:
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
            self.c1 = tf.layers.conv1d(self.rs, input_dim*2, 4, padding = 'same',activation=tf.nn.relu ,name='conv1d1')
            self.c2 = tf.layers.conv1d(self.c1, input_dim*2, 4, padding = 'same',activation=tf.nn.relu ,name='conv1d2')
            self.rs2 = tf.reshape(self.c2, [-1, input_dim*2*input_history],name='output')
            self.d1 = tf.layers.dense(self.rs2, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 128, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)

class GraphCNN_k5c2ik5c2id128d128:
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
            self.c1 = tf.layers.conv1d(self.rs, input_dim*2, 5, padding = 'same',activation=tf.nn.relu ,name='conv1d1')
            self.c2 = tf.layers.conv1d(self.c1, input_dim*2, 5, padding = 'same',activation=tf.nn.relu ,name='conv1d2')
            self.rs2 = tf.reshape(self.c2, [-1, input_dim*2*input_history],name='output')
            self.d1 = tf.layers.dense(self.rs2, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 128, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)

class GraphCNN_k6c2ik6c2id128d128:
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
            self.c1 = tf.layers.conv1d(self.rs, input_dim*2, 4, padding = 'same',activation=tf.nn.relu ,name='conv1d1')
            self.c2 = tf.layers.conv1d(self.c1, input_dim*2, 4, padding = 'same',activation=tf.nn.relu ,name='conv1d2')
            self.rs2 = tf.reshape(self.c2, [-1, input_dim*2*input_history],name='output')
            self.d1 = tf.layers.dense(self.rs2, 128, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 128, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # Loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)

class GraphCNN_k3c2ik3c2id64d32:
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
            self.c1 = tf.layers.conv1d(self.rs, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1')
            self.c2 = tf.layers.conv1d(self.c1, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d2')
            self.rs2 = tf.reshape(self.c2, [-1, input_dim*2*input_history],name='output')
            self.d1 = tf.layers.dense(self.rs2, 64, activation=tf.nn.relu, name='dense1')
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

class GraphCNN_k4c2ik4c2id64d32:
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
            self.c1 = tf.layers.conv1d(self.rs, input_dim*2, 4, padding = 'same',activation=tf.nn.relu ,name='conv1d1')
            self.c2 = tf.layers.conv1d(self.c1, input_dim*2, 4, padding = 'same',activation=tf.nn.relu ,name='conv1d2')
            self.rs2 = tf.reshape(self.c2, [-1, input_dim*2*input_history],name='output')
            self.d1 = tf.layers.dense(self.rs2, 64, activation=tf.nn.relu, name='dense1')
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

class GraphCNN_k5c2ik5c2id64d32:
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
            self.c1 = tf.layers.conv1d(self.rs, input_dim*2, 5, padding = 'same',activation=tf.nn.relu ,name='conv1d1')
            self.c2 = tf.layers.conv1d(self.c1, input_dim*2, 5, padding = 'same',activation=tf.nn.relu ,name='conv1d2')
            self.rs2 = tf.reshape(self.c2, [-1, input_dim*2*input_history],name='output')
            self.d1 = tf.layers.dense(self.rs2, 64, activation=tf.nn.relu, name='dense1')
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

class GraphCNN_k6c2ik6c2id64d32:
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
            self.c1 = tf.layers.conv1d(self.rs, input_dim*2, 4, padding = 'same',activation=tf.nn.relu ,name='conv1d1')
            self.c2 = tf.layers.conv1d(self.c1, input_dim*2, 4, padding = 'same',activation=tf.nn.relu ,name='conv1d2')
            self.rs2 = tf.reshape(self.c2, [-1, input_dim*2*input_history],name='output')
            self.d1 = tf.layers.dense(self.rs2, 64, activation=tf.nn.relu, name='dense1')
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

class GraphCNN_k3c2ik3c2ik3c2ik3c2id64d64:
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
            self.c1 = tf.layers.conv1d(self.rs, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1')
            self.c2 = tf.layers.conv1d(self.c1, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d2')
            self.c3 = tf.layers.conv1d(self.c2, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d3')
            self.c4 = tf.layers.conv1d(self.c3, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4')
            self.rs2 = tf.reshape(self.c4, [-1, input_dim*2*input_history],name='output')
            self.d1 = tf.layers.dense(self.rs2, 64, activation=tf.nn.relu, name='dense1')
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

class GraphCNN_k3c2ik3c2ip2k3c2ik3c2id64d64:
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
            self.c1 = tf.layers.conv1d(self.rs, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1')
            self.c2 = tf.layers.conv1d(self.c1, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d2')
            self.p1 = tf.layers.max_pooling1d(self.c2, 2, 2, padding='same', name='pool1d1')
            self.c3 = tf.layers.conv1d(self.p1, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d3')
            self.c4 = tf.layers.conv1d(self.c3, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4')
            self.rs2 = tf.reshape(self.c4, [-1, input_dim*input_history],name='output')
            self.d1 = tf.layers.dense(self.rs2, 64, activation=tf.nn.relu, name='dense1')
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

class GraphCNN_k3c2ik3c2ip2k3c2ik3c2ip2d64d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions

        self.g = tf.Graph()
        with self.g.as_default():

            # placeholders
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # operations
            self.c1 = tf.layers.conv1d(self.rs, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1')
            self.c2 = tf.layers.conv1d(self.c1, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d2')
            self.p1 = tf.layers.max_pooling1d(self.c2, 2, 2, padding='same', name='pool1d1')
            self.c3 = tf.layers.conv1d(self.p1, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d3')
            self.c4 = tf.layers.conv1d(self.c3, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4')
            self.p2 = tf.layers.max_pooling1d(self.c4, 2, 2, padding='same', name='pool1d2')
            self.rs2 = tf.reshape(self.p2, [-1, int(input_dim*input_history/2)],name='output')
            self.d1 = tf.layers.dense(self.rs2, 64, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)

class GraphCNN_k5c2ip2k5c2id64d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions

        self.g = tf.Graph()
        with self.g.as_default():

            # placeholders
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # operations
            self.c1 = tf.layers.conv1d(self.rs, input_dim*2, 5, padding = 'same',activation=tf.nn.relu ,name='conv1d1')
            self.p1 = tf.layers.max_pooling1d(self.c1, 2, 2, padding='same', name='pool1d1')
            self.c2 = tf.layers.conv1d(self.p1, input_dim*2, 5, padding = 'same',activation=tf.nn.relu ,name='conv1d2')
            self.rs2 = tf.reshape(self.c2, [-1, input_dim*input_history],name='rs')
            self.d1 = tf.layers.dense(self.rs2, 64, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)

class GraphCNN_k5c2ip2k5c2ip2d64d64:
    def __init__(self,input_history,input_dim,output_forecast,output_dim):
        #inspired from https://stackoverflow.com/questions/44418442/building-tensorflow-graphs-inside-of-functions

        self.g = tf.Graph()
        with self.g.as_default():

            # placeholders
            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * output_dim], name='target')
            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')
            self.rs = tf.reshape(self.x, [-1, input_history, input_dim],name='reshape')

            # operations
            self.c1 = tf.layers.conv1d(self.rs, input_dim*2, 5, padding = 'same',activation=tf.nn.relu ,name='conv1d1')
            self.p1 = tf.layers.max_pooling1d(self.c1, 2, 2, padding='same', name='pool1d1')
            self.c2 = tf.layers.conv1d(self.p1, input_dim*2, 5, padding = 'same',activation=tf.nn.relu ,name='conv1d2')
            self.p2 = tf.layers.max_pooling1d(self.c2, 2, 2, padding='same', name='pool1d2')
            self.rs2 = tf.reshape(self.p2, [-1, int(input_dim*input_history/2)],name='rs')
            self.d1 = tf.layers.dense(self.rs2, 64, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 64, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')

            # loss
            self.diff = tf.square(tf.subtract(self.y_,self.y))
            self.s_loss = tf.reduce_mean(self.diff, axis=1)
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # train
            self.grad = tf.norm(tf.gradients(self.s_loss, self.y_),axis=2)
            self.acc_op = accuracy(self.diff)
            self.train_step = train_fn(self.w_loss, 0.01)

class GraphCNN_k3cik3cid64d64:
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
            self.c1 = tf.layers.conv1d(self.rs, input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1')
            self.c2 = tf.layers.conv1d(self.c1, input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d2')
            self.rs2 = tf.reshape(self.c2, [-1, input_dim*input_history],name='rs')
            self.d1 = tf.layers.dense(self.rs2, 64, activation=tf.nn.relu, name='dense1')
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
class GraphCNN_k3c3ik3c3id64d64:
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
            self.c1 = tf.layers.conv1d(self.rs, input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1')
            self.c2 = tf.layers.conv1d(self.c1, input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d2')
            self.rs2 = tf.reshape(self.c2, [-1, input_dim*input_history*3],name='rs')
            self.d1 = tf.layers.dense(self.rs2, 64, activation=tf.nn.relu, name='dense1')
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
class GraphCNN_k3c4ik3c4id64d64:
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
            self.c1 = tf.layers.conv1d(self.rs, input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1')
            self.c2 = tf.layers.conv1d(self.c1, input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d2')
            self.rs2 = tf.reshape(self.c2, [-1, input_dim*input_history*4],name='rs')
            self.d1 = tf.layers.dense(self.rs2, 64, activation=tf.nn.relu, name='dense1')
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
class GraphCNN_k3cik3cid128d64:
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
            self.c1 = tf.layers.conv1d(self.rs, input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1')
            self.c2 = tf.layers.conv1d(self.c1, input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d2')
            self.rs2 = tf.reshape(self.c2, [-1, input_dim*input_history],name='rs')
            self.d1 = tf.layers.dense(self.rs2, 128, activation=tf.nn.relu, name='dense1')
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
class GraphCNN_k3c2ik3c2id128d64:
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
            self.c1 = tf.layers.conv1d(self.rs, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1')
            self.c2 = tf.layers.conv1d(self.c1, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d2')
            self.rs2 = tf.reshape(self.c2, [-1, input_dim*input_history*2],name='rs')
            self.d1 = tf.layers.dense(self.rs2, 128, activation=tf.nn.relu, name='dense1')
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
class GraphCNN_k3c3ik3c3id128d64:
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
            self.c1 = tf.layers.conv1d(self.rs, input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1')
            self.c2 = tf.layers.conv1d(self.c1, input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d2')
            self.rs2 = tf.reshape(self.c2, [-1, input_dim*input_history*3],name='rs')
            self.d1 = tf.layers.dense(self.rs2, 128, activation=tf.nn.relu, name='dense1')
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
class GraphCNN_k3c4ik3c4id128d64:
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
            self.c1 = tf.layers.conv1d(self.rs, input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1')
            self.c2 = tf.layers.conv1d(self.c1, input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d2')
            self.rs2 = tf.reshape(self.c2, [-1, input_dim*input_history*4],name='rs')
            self.d1 = tf.layers.dense(self.rs2, 128, activation=tf.nn.relu, name='dense1')
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
class GraphCNN_k3cik3cid128d32:
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
            self.c1 = tf.layers.conv1d(self.rs, input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1')
            self.c2 = tf.layers.conv1d(self.c1, input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d2')
            self.rs2 = tf.reshape(self.c2, [-1, input_dim*input_history],name='rs')
            self.d1 = tf.layers.dense(self.rs2, 128, activation=tf.nn.relu, name='dense1')
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
class GraphCNN_k3c2ik3c2id128d32:
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
            self.c1 = tf.layers.conv1d(self.rs, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1')
            self.c2 = tf.layers.conv1d(self.c1, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d2')
            self.rs2 = tf.reshape(self.c2, [-1, input_dim*input_history*2],name='rs')
            self.d1 = tf.layers.dense(self.rs2, 128, activation=tf.nn.relu, name='dense1')
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
class GraphCNN_k3c3ik3c3id128d32:
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
            self.c1 = tf.layers.conv1d(self.rs, input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1')
            self.c2 = tf.layers.conv1d(self.c1, input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d2')
            self.rs2 = tf.reshape(self.c2, [-1, input_dim*input_history*3],name='rs')
            self.d1 = tf.layers.dense(self.rs2, 128, activation=tf.nn.relu, name='dense1')
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
class GraphCNN_k3c4ik3c4id128d32:
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
            self.c1 = tf.layers.conv1d(self.rs, input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1')
            self.c2 = tf.layers.conv1d(self.c1, input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d2')
            self.rs2 = tf.reshape(self.c2, [-1, input_dim*input_history*4],name='rs')
            self.d1 = tf.layers.dense(self.rs2, 128, activation=tf.nn.relu, name='dense1')
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
class GraphCNN_k3c4ik3c4id128d32:
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
            self.c1 = tf.layers.conv1d(self.rs, input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1')
            self.c2 = tf.layers.conv1d(self.c1, input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d2')
            self.rs2 = tf.reshape(self.c2, [-1, input_dim*input_history*4],name='rs')
            self.d1 = tf.layers.dense(self.rs2, 128, activation=tf.nn.relu, name='dense1')
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
class GraphCNN_k3cik3cip2k3c2ik3c2id64d64:
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
            self.c1 = tf.layers.conv1d(self.rs, input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1')
            self.c2 = tf.layers.conv1d(self.c1, input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d2')
            self.p1 = tf.layers.max_pooling1d(self.c2, 2, 2, padding='same', name='pool1d1')
            self.c3 = tf.layers.conv1d(self.p1, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d3')
            self.c4 = tf.layers.conv1d(self.c3, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4')
            self.rs2 = tf.reshape(self.c4, [-1, input_dim*input_history],name='rs')
            self.d1 = tf.layers.dense(self.rs2, 64, activation=tf.nn.relu, name='dense1')
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
class GraphCNN_k3c2ik3c2ip2k3c4ik3c4id64d64:
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
            self.c1 = tf.layers.conv1d(self.rs, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1')
            self.c2 = tf.layers.conv1d(self.c1, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d2')
            self.p1 = tf.layers.max_pooling1d(self.c2, 2, 2, padding='same', name='pool1d1')
            self.c3 = tf.layers.conv1d(self.p1, input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d3')
            self.c4 = tf.layers.conv1d(self.c3, input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4')
            self.rs2 = tf.reshape(self.c4, [-1, input_dim*input_history*2],name='rs')
            self.d1 = tf.layers.dense(self.rs2, 64, activation=tf.nn.relu, name='dense1')
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
class GraphCNN_k3c3ik3c3ip2k3c6ik3c6id64d64:
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
            self.c1 = tf.layers.conv1d(self.rs, input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1')
            self.c2 = tf.layers.conv1d(self.c1, input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d2')
            self.p1 = tf.layers.max_pooling1d(self.c2, 2, 2, padding='same', name='pool1d1')
            self.c3 = tf.layers.conv1d(self.p1, input_dim*6, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d3')
            self.c4 = tf.layers.conv1d(self.c3, input_dim*6, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4')
            self.rs2 = tf.reshape(self.c4, [-1, input_dim*input_history*3],name='rs')
            self.d1 = tf.layers.dense(self.rs2, 64, activation=tf.nn.relu, name='dense1')
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
class GraphCNN_k3cik3cip2k3c2ik3c2ip2d64d64:
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
            self.c1 = tf.layers.conv1d(self.rs, input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1')
            self.c2 = tf.layers.conv1d(self.c1, input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d2')
            self.p1 = tf.layers.max_pooling1d(self.c2, 2, 2, padding='same', name='pool1d1')
            self.c3 = tf.layers.conv1d(self.p1, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d3')
            self.c4 = tf.layers.conv1d(self.c3, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4')
            self.p2 = tf.layers.max_pooling1d(self.c4, 2, 2, padding='same', name='pool1d2')
            self.rs2 = tf.reshape(self.p2, [-1, int(input_dim*input_history/2)],name='rs')
            self.d1 = tf.layers.dense(self.rs2, 64, activation=tf.nn.relu, name='dense1')
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
class GraphCNN_k3c2ik3c2ip2k3c4ik3c4ip2d64d64:
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
            self.c1 = tf.layers.conv1d(self.rs, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1')
            self.c2 = tf.layers.conv1d(self.c1, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d2')
            self.p1 = tf.layers.max_pooling1d(self.c2, 2, 2, padding='same', name='pool1d1')
            self.c3 = tf.layers.conv1d(self.p1, input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d3')
            self.c4 = tf.layers.conv1d(self.c3, input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4')
            self.p2 = tf.layers.max_pooling1d(self.c4, 2, 2, padding='same', name='pool1d2')
            self.rs2 = tf.reshape(self.p2, [-1, input_dim*input_history],name='rs')
            self.d1 = tf.layers.dense(self.rs2, 64, activation=tf.nn.relu, name='dense1')
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
class GraphCNN_k3c3ik3c3ip2k3c6ik3c6ip2d64d64:
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
            self.c1 = tf.layers.conv1d(self.rs, input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1')
            self.c2 = tf.layers.conv1d(self.c1, input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d2')
            self.p1 = tf.layers.max_pooling1d(self.c2, 2, 2, padding='same', name='pool1d1')
            self.c3 = tf.layers.conv1d(self.p1, input_dim*6, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d3')
            self.c4 = tf.layers.conv1d(self.c3, input_dim*6, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4')
            self.p2 = tf.layers.max_pooling1d(self.c4, 2, 2, padding='same', name='pool1d2')
            self.rs2 = tf.reshape(self.p2, [-1, int(input_dim*input_history*3/2)],name='rs')
            self.d1 = tf.layers.dense(self.rs2, 64, activation=tf.nn.relu, name='dense1')
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
class GraphCNN_k3cik3cip2k3c2ik3c2id64d32:
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
            self.c1 = tf.layers.conv1d(self.rs, input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1')
            self.c2 = tf.layers.conv1d(self.c1, input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d2')
            self.p1 = tf.layers.max_pooling1d(self.c2, 2, 2, padding='same', name='pool1d1')
            self.c3 = tf.layers.conv1d(self.p1, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d3')
            self.c4 = tf.layers.conv1d(self.c3, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4')
            self.rs2 = tf.reshape(self.c4, [-1, input_dim*input_history],name='rs')
            self.d1 = tf.layers.dense(self.rs2, 64, activation=tf.nn.relu, name='dense1')
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
class GraphCNN_k3c2ik3c2ip2k3c4ik3c4id64d32:
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
            self.c1 = tf.layers.conv1d(self.rs, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1')
            self.c2 = tf.layers.conv1d(self.c1, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d2')
            self.p1 = tf.layers.max_pooling1d(self.c2, 2, 2, padding='same', name='pool1d1')
            self.c3 = tf.layers.conv1d(self.p1, input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d3')
            self.c4 = tf.layers.conv1d(self.c3, input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4')
            self.rs2 = tf.reshape(self.c4, [-1, input_dim*input_history*2],name='rs')
            self.d1 = tf.layers.dense(self.rs2, 64, activation=tf.nn.relu, name='dense1')
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
class GraphCNN_k3c3ik3c3ip2k3c6ik3c6id64d32:
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
            self.c1 = tf.layers.conv1d(self.rs, input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1')
            self.c2 = tf.layers.conv1d(self.c1, input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d2')
            self.p1 = tf.layers.max_pooling1d(self.c2, 2, 2, padding='same', name='pool1d1')
            self.c3 = tf.layers.conv1d(self.p1, input_dim*6, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d3')
            self.c4 = tf.layers.conv1d(self.c3, input_dim*6, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4')
            self.rs2 = tf.reshape(self.c4, [-1, input_dim*input_history*3],name='rs')
            self.d1 = tf.layers.dense(self.rs2, 64, activation=tf.nn.relu, name='dense1')
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
class GraphCNN_k3cik3cip2k3c2ik3c2ip2d64d32:
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
            self.c1 = tf.layers.conv1d(self.rs, input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1')
            self.c2 = tf.layers.conv1d(self.c1, input_dim, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d2')
            self.p1 = tf.layers.max_pooling1d(self.c2, 2, 2, padding='same', name='pool1d1')
            self.c3 = tf.layers.conv1d(self.p1, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d3')
            self.c4 = tf.layers.conv1d(self.c3, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4')
            self.p2 = tf.layers.max_pooling1d(self.c4, 2, 2, padding='same', name='pool1d2')
            self.rs2 = tf.reshape(self.p2, [-1, int(input_dim*input_history/2)],name='rs')
            self.d1 = tf.layers.dense(self.rs2, 64, activation=tf.nn.relu, name='dense1')
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
class GraphCNN_k3c2ik3c2ip2k3c4ik3c4ip2d64d32:
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
            self.c1 = tf.layers.conv1d(self.rs, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1')
            self.c2 = tf.layers.conv1d(self.c1, input_dim*2, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d2')
            self.p1 = tf.layers.max_pooling1d(self.c2, 2, 2, padding='same', name='pool1d1')
            self.c3 = tf.layers.conv1d(self.p1, input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d3')
            self.c4 = tf.layers.conv1d(self.c3, input_dim*4, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4')
            self.p2 = tf.layers.max_pooling1d(self.c4, 2, 2, padding='same', name='pool1d2')
            self.rs2 = tf.reshape(self.p2, [-1, input_dim*input_history],name='rs')
            self.d1 = tf.layers.dense(self.rs2, 64, activation=tf.nn.relu, name='dense1')
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
class GraphCNN_k3c3ik3c3ip2k3c6ik3c6ip2d64d32:
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
            self.c1 = tf.layers.conv1d(self.rs, input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d1')
            self.c2 = tf.layers.conv1d(self.c1, input_dim*3, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d2')
            self.p1 = tf.layers.max_pooling1d(self.c2, 2, 2, padding='same', name='pool1d1')
            self.c3 = tf.layers.conv1d(self.p1, input_dim*6, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d3')
            self.c4 = tf.layers.conv1d(self.c3, input_dim*6, 3, padding = 'same',activation=tf.nn.relu ,name='conv1d4')
            self.p2 = tf.layers.max_pooling1d(self.c4, 2, 2, padding='same', name='pool1d2')
            self.rs2 = tf.reshape(self.p2, [-1, int(input_dim*input_history*3/2)],name='rs')
            self.d1 = tf.layers.dense(self.rs2, 64, activation=tf.nn.relu, name='dense1')
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

class GraphLoss:

    def __init__(self,input_history,input_dim,output_forecast,output_dim,cmd_dim):


        # @TODO : generalisation at N step forward (instead of current hardoded 5)
        self.g = tf.Graph()
        with self.g.as_default():

            # PLACEHOLDERS

            self.x = tf.placeholder(tf.float32, shape=[None, input_history * input_dim], name='inputs')
            self.y = tf.placeholder(tf.float32, shape=[None, output_forecast * input_dim], name='target')# need input dimention for rebuilding input state (input : state + cmd , ouptut : state


            self.step = tf.placeholder(tf.int32)
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.weights = tf.placeholder(tf.float32, shape=[None], name='weights')

            # Operations
            # @TODO : managing cmd/output better for generalisation and comprehension

            self.d1 = tf.layers.dense(self.x, 32, activation=tf.nn.relu, name='dense1')
            self.d2 = tf.layers.dense(self.d1, 32, activation=tf.nn.relu, name='dense2')
            self.y_ = tf.layers.dense(self.d2, output_dim, activation=None, name='output')
            #print(self.y_.shape)
            self.tmp_cmd0 = self.y[:,(output_forecast - 4) * (output_dim + cmd_dim) - cmd_dim :(output_forecast - 4) * (output_dim + cmd_dim)]
            #print(self.tmp_cmd0)
            self.x1 = tf.concat((tf.concat((self.y_, self.tmp_cmd0), axis=1), self.x[:,:-input_dim]),axis=1)
            #print(self.x1.shape)
            self.d11 = tf.layers.dense(self.x1, 32, activation=tf.nn.relu, name='dense1', reuse=True)
            self.d12 = tf.layers.dense(self.d11, 32, activation=tf.nn.relu, name='dense2', reuse=True)
            self.y1_ = tf.layers.dense(self.d12, output_dim, activation=None, name='output', reuse=True)

            self.tmp_cmd1 = self.y[:,(output_forecast - 3) * (output_dim + cmd_dim) - cmd_dim :(output_forecast - 3) * (output_dim + cmd_dim)]
            self.x2 = tf.concat((tf.concat((self.y1_, self.tmp_cmd1), axis=1), self.x1[:,:-input_dim]), axis=1)

            self.d21 = tf.layers.dense(self.x2, 32, activation=tf.nn.relu, name='dense1', reuse=True)
            self.d22 = tf.layers.dense(self.d21, 32, activation=tf.nn.relu, name='dense2', reuse=True)
            self.y2_ = tf.layers.dense(self.d22, output_dim, activation=None, name='output', reuse=True)

            self.tmp_cmd2 = self.y[:,(output_forecast - 2) * (output_dim + cmd_dim) - cmd_dim :(output_forecast -2) * (output_dim + cmd_dim)]
            self.x3 = tf.concat((tf.concat((self.y2_, self.tmp_cmd2), axis=1), self.x2[:,:-input_dim]), axis=1)

            self.d31 = tf.layers.dense(self.x3, 32, activation=tf.nn.relu, name='dense1', reuse=True)
            self.d32 = tf.layers.dense(self.d31, 32, activation=tf.nn.relu, name='dense2', reuse=True)
            self.y3_ = tf.layers.dense(self.d32, output_dim, activation=None, name='output', reuse=True)

            self.tmp_cmd3 = self.y[:, (output_forecast-1) * (output_dim + cmd_dim) - cmd_dim: (output_forecast-1) * (output_dim + cmd_dim)]#assume y [state cmd [...] state cmd] # taking the last cmd
            self.x4 = tf.concat((tf.concat((self.y3_,self.tmp_cmd3),axis=1), self.x3[:,:-input_dim]), axis=1)

            self.d41 = tf.layers.dense(self.x4, 32, activation=tf.nn.relu, name='dense1', reuse=True)
            self.d42 = tf.layers.dense(self.d41, 32, activation=tf.nn.relu, name='dense2', reuse=True)
            self.y4_ = tf.layers.dense(self.d42, output_dim, activation=None, name='output', reuse=True)

            # Loss

            self.tmp_y4 = self.y[:, (output_forecast - 1) * (output_dim + cmd_dim): output_forecast * (output_dim + cmd_dim) - cmd_dim]#assume y [state cmd [...] state cmd] # taking the last cmd
            self.tmp_y3 = self.y[:, (output_forecast - 2) * (output_dim + cmd_dim): (output_forecast - 1) * (output_dim + cmd_dim) - cmd_dim]#assume y [state cmd [...] state cmd] # taking the last cmd
            self.tmp_y2 = self.y[:, (output_forecast - 3) * (output_dim + cmd_dim): (output_forecast - 2) * (output_dim + cmd_dim) - cmd_dim]#assume y [state cmd [...] state cmd] # taking the last cmd
            self.tmp_y1 = self.y[:, (output_forecast - 4) * (output_dim + cmd_dim): (output_forecast - 3) * (output_dim + cmd_dim) - cmd_dim]#assume y [state cmd [...] state cmd] # taking the last cmd
            self.tmp_y = self.y[:,:(output_dim + cmd_dim) - cmd_dim]#assume y [state cmd [...] state cmd] # taking the last cmd output_forecast-5 = 0 ; outputforecast == n in loss on n order

            # @TODO fully put order into parameter

            self.diff0 = tf.reduce_mean(tf.square(tf.subtract(self.y_,self.tmp_y)),axis=1)
            self.diff1 = tf.reduce_mean(tf.square(tf.subtract(self.y1_,self.tmp_y1)),axis=1)
            self.diff2 = tf.reduce_mean(tf.square(tf.subtract(self.y2_,self.tmp_y2)),axis=1)
            self.diff3 = tf.reduce_mean(tf.square(tf.subtract(self.y3_,self.tmp_y3)),axis=1)
            self.diff4 = tf.reduce_mean(tf.square(tf.subtract(self.y4_,self.tmp_y4)),axis=1)

            self.s_loss = self.diff0# + self.diff1 + self.diff2 + self.diff3 + self.diff4
            #self.s_loss = tf.reduce_logsumexp([self.diff0,self.diff1,self.diff2,self.diff3,self.diff4])
            self.w_loss = tf.reduce_mean(tf.multiply(self.s_loss, self.weights))
            # Train

            self.grad0 = tf.norm(tf.gradients(self.diff0, self.y_),axis=2)
            self.grad1 = tf.norm(tf.gradients(self.diff1, self.y1_),axis=2)
            self.grad2 = tf.norm(tf.gradients(self.diff2, self.y2_),axis=2)
            self.grad3 = tf.norm(tf.gradients(self.diff3, self.y3_),axis=2)
            self.grad4 = tf.norm(tf.gradients(self.diff4, self.y4_),axis=2)
            self.grads = self.grad0 + self.grad1 + self.grad2 + self.grad3 + self.grad4

            self.loss_acc = tf.reduce_mean(self.diff0)
            self.acc_op = accuracy(self.loss_acc)
            self.train_step = train_fn(self.w_loss, 0.01)

