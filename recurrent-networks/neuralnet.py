import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

class DynamicLSTM:
    def  __init__(self, settings):
        # Quick Setup
        self.settings = settings
        self.loss_type = None

        # Placeholders
        self.BX = None
        self.BY = None
        self.HS = None
        self.W  = None

        # RNN variables
        self.rnn_tuple_state = None
        self.state_series = None
        self.current_state = None
        self.losses = None
        self.reset_state = None
        
        # Net operations
        self.losses = None
        self.loss = None
        self.train_step = None
        self.accuracy = None

        # Tensorboard
        self.summaries = None

        # Run
        self.selector()

    def selector(self):
        if self.settings.priorization == 'PER':
            if self.settings.rnn_model == 'rnn-basic':
                self.rnn_basic_per()
            elif self.settings.rnn_model == 'lstm-basic':
                self.lstm_basic_per()
                self.reset_state = np.zeros((self.settings.num_layers, 2, self.settings.batch_size, self.settings.state_size))
            elif self.settings.rnn_model == 'gru-basic':
                self.gru_basic_per()
        else:
            ###### RNNs ######
            if self.settings.rnn_model == 'rnn_hsX_lX_m1':
                self.rnn_hsX_lX_m1()
                self.reset_state = np.zeros((self.settings.num_layers, self.settings.batch_size, self.settings.state_size))
            elif self.settings.rnn_model == 'rnn_hsX_lX_m2_16':
                self.rnn_hsX_lX_m1()
                self.reset_state = np.zeros((self.settings.num_layers, self.settings.batch_size, self.settings.state_size))
            elif self.settings.rnn_model == 'rnn_hsX_lX_m2_32':
                self.rnn_hsX_lX_m1()
                self.reset_state = np.zeros((self.settings.num_layers, self.settings.batch_size, self.settings.state_size))
            elif self.settings.rnn_model == 'rnn_hsX_lX_m2_64':
                self.rnn_hsX_lX_m1()
                self.reset_state = np.zeros((self.settings.num_layers, self.settings.batch_size, self.settings.state_size))
            ###### GRUs ######
            elif self.settings.rnn_model == 'gru_hsX_lX_m1':
                self.gru_hsX_lX_m1()
                self.reset_state = np.zeros((self.settings.num_layers, self.settings.batch_size, self.settings.state_size))
            elif self.settings.rnn_model == 'gru_hsX_lX_m2_16':
                self.gru_hsX_lX_m2_16()
                self.reset_state = np.zeros((self.settings.num_layers, self.settings.batch_size, self.settings.state_size))
            elif self.settings.rnn_model == 'gru_hsX_lX_m2_32':
                self.gru_hsX_lX_m2_32()
                self.reset_state = np.zeros((self.settings.num_layers, self.settings.batch_size, self.settings.state_size))
            elif self.settings.rnn_model == 'gru_hsX_lX_m2_64':
                self.gru_hsX_lX_m2_64()
                self.reset_state = np.zeros((self.settings.num_layers, self.settings.batch_size, self.settings.state_size))
            ###### LSTMs ######
            elif self.settings.rnn_model == 'lstm_hsX_lX_m1':
                self.lstm_hsX_lX_m1()
                self.reset_state = np.zeros((self.settings.num_layers, 2, self.settings.batch_size, self.settings.state_size))
            elif self.settings.rnn_model == 'lstm_hsX_lX_m2_16':
                self.lstm_hsX_lX_m2_l16()
                self.reset_state = np.zeros((self.settings.num_layers, 2, self.settings.batch_size, self.settings.state_size))
            elif self.settings.rnn_model == 'lstm_hsX_lX_m2_32':
                self.lstm_hsX_lX_m2_l32()
                self.reset_state = np.zeros((self.settings.num_layers, 2, self.settings.batch_size, self.settings.state_size))
            elif self.settings.rnn_model == 'lstm_hsX_lX_m2_64':
                self.lstm_hsX_lX_m2_l64()
                self.reset_state = np.zeros((self.settings.num_layers, 2, self.settings.batch_size, self.settings.state_size))

    def tensorboard(self):
        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.scalar('loss', self.loss)
        self.summaries = tf.summary.merge_all()

    def lstm_placeholders(self):
        self.BX = tf.placeholder(tf.float32, [self.settings.batch_size, self.settings.sequence_length, self.settings.input_dim], name = 'input')
        self.BY = tf.placeholder(tf.float32, [self.settings.batch_size, self.settings.sequence_length, self.settings.output_dim], name = 'target')
        self.HS = tf.placeholder(tf.float32, [self.settings.num_layers, 2, self.settings.batch_size, self.settings.state_size], name = 'hidden_state')
        self.W = tf.placeholder(tf.float32, [self.settings.batch_size*self.settings.sequence_length], name = 'priorization_weights')
    
    def rnn_placeholders(self):
        self.BX = tf.placeholder(tf.float32, [self.settings.batch_size, self.settings.sequence_length, self.settings.input_dim], name = 'input')
        self.BY = tf.placeholder(tf.float32, [self.settings.batch_size, self.settings.sequence_length, self.settings.output_dim], name = 'target')
        self.HS = tf.placeholder(tf.float32, [self.settings.num_layers, self.settings.batch_size, self.settings.state_size], name = 'hidden_state')
        self.W = tf.placeholder(tf.float32, [self.settings.batch_size*self.settings.sequence_length], name = 'priorization_weights')

    def lstm_hidden_state(self):
        state_per_layer_list = tf.unstack(self.HS, axis=0)
        self.rnn_tuple_state = tuple(
            [tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
            for idx in range(self.settings.num_layers)]
            )
    
    def rnn_hidden_state(self):
        #state_per_layer_list = tf.unstack(self.HS, axis=0)
        self.rnn_tuple_state = tuple([self.HS[idx] for idx in range(self.settings.num_layers)])
            #[tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
            #for idx in range(self.settings.num_layers)]
            #)

    def rnn_basic_per(self):
        raise Exception('Not implemented')

    def lstm_basic_per(self):
        # Setup
        self.lstm_placeholders()
        self.lstm_hidden_state()

        # Forward pass
        stacked_rnn = []
        for _ in range(self.settings.num_layers):
            stacked_rnn.append(tf.nn.rnn_cell.LSTMCell(self.settings.state_size, state_is_tuple=True, name='lstm_cell_'+str(_)))

        cell = tf.nn.rnn_cell.MultiRNNCell(stacked_rnn,
                state_is_tuple = True)
        self.states_series, self.current_state = tf.nn.dynamic_rnn(cell, self.BX, initial_state=self.rnn_tuple_state)
        self.states_series = tf.reshape(self.states_series, [-1, self.settings.state_size], name = 'reshape_state')
        with tf.name_scope('dense'):
            W2 = tf.Variable(np.random.rand(self.settings.state_size, self.settings.output_dim),dtype=tf.float32, name = 'weights')
            b2 = tf.Variable(np.zeros((1,self.settings.output_dim)), dtype=tf.float32, name = 'bias')
            self.logits = tf.matmul(self.states_series, W2) + b2
        
        # Operations
        self.labels = tf.reshape(self.BY, [-1, self.settings.output_dim])
        self.losses = tf.losses.huber_loss(self.labels, self.logits, reduction = tf.losses.Reduction.NONE)
        with tf.name_scope('losses'):
            self.loss = tf.reduce_mean(self.losses)
        with tf.name_scope('weighted_losses'):
            self.weighted_loss = tf.reduce_mean(tf.reduce_mean(self.losses,axis=1)*self.W)
        with tf.name_scope('accuracy_op'):
            self.accuracy = tf.reduce_mean(tf.square(tf.subtract(self.labels, self.logits)))
        with tf.name_scope('train_step'):
            self.train_step = tf.train.AdamOptimizer(self.settings.learning_rate).minimize(self.weighted_loss)
        
        # Tensorboard
        self.tensorboard()

    def gru_basic_pe(self):
        raise Exception('Not implemented')
    def rnn_adv_1_per(self):
        raise Exception('Not implemented')
    def lstm_adv_1_per(self):
        raise Exception('Not implemented')
    def gru_adv_1_per(self):
        raise Exception('Not implemented')
    def rnn_basic(self):
        raise Exception('Not implemented')

    def lstm_basic(self):
        # Setup
        self.lstm_placeholders()
        self.lstm_hidden_state()

        # Forward pass
        stacked_rnn = []
        for _ in range(self.settings.num_layers):
            stacked_rnn.append(tf.nn.rnn_cell.LSTMCell(self.settings.state_size, state_is_tuple=True, name='lstm_cell_'+str(_)))

        cell = tf.nn.rnn_cell.MultiRNNCell(stacked_rnn, state_is_tuple = True)

        # Feeds a list to the fuck bag
        self.BXR = tf.reshape(self.BX, [-1, self.settings.sequence_length*self.settings.input_dim], name="BX-R")
        self.BYR = tf.reshape(self.BY, [-1, self.settings.sequence_length*self.settings.output_dim], name="BY-R")
        self.inputs_series = tf.split(self.BXR, self.settings.sequence_length, axis=1)
        self.labels_series = tf.split(self.BYR, self.settings.sequence_length, axis=1)

        # Nasty'ol trick to get the current state of the lstm after the first
        # run. This allows us to perform stridding predictions.
        self.input_one = [self.inputs_series[0]]
        self.input_two = self.inputs_series[1:]
        self.first_series, self.mid_state = tf.nn.static_rnn(cell, self.input_one, initial_state=self.rnn_tuple_state)
        self.second_series, self.current_state = tf.nn.static_rnn(cell, self.input_two, initial_state=self.mid_state)
        self.states_series = self.first_series + self.second_series

        with tf.name_scope('dense'):
            W2 = tf.Variable(np.random.rand(self.settings.state_size, self.settings.output_dim),dtype=tf.float32, name = 'weights')
            b2 = tf.Variable(np.zeros((1,self.settings.output_dim)), dtype=tf.float32, name = 'bias')
            self.logits_series = [tf.matmul(state, W2) + b2 for state in self.states_series]
        self.losses = [tf.losses.huber_loss(logits,labels) for logits, labels in zip(self.logits_series,self.labels_series)]
        #self.losses = [tf.square(tf.subtract(logits,labels)) for logits, labels in zip(self.logits_series,self.labels_series)]

        # Probably would be better to rename all the appearances of logits to
        # logits series.
        self.logits = self.logits_series

        # Operations
        with tf.name_scope('losses'):
            self.loss = tf.reduce_mean(self.losses)
        with tf.name_scope('accuracy_op'):
            self.accuracy = tf.reduce_mean([tf.square(tf.subtract(labels, logits)) for logits, labels in zip(self.logits_series, self.labels_series)])
        with tf.name_scope('train_step'):
            self.train_step = tf.train.AdamOptimizer(self.settings.learning_rate).minimize(self.loss)
        
        # Tensorboard
        self.tensorboard()

    def lstm_basic_dyn(self):
        # Setup
        self.lstm_placeholders()
        self.lstm_hidden_state()

        # Forward pass
        stacked_rnn = []
        for _ in range(self.settings.num_layers):
            stacked_rnn.append(tf.nn.rnn_cell.LSTMCell(self.settings.state_size, state_is_tuple=True, name='lstm_cell_'+str(_)))

        cell = tf.nn.rnn_cell.MultiRNNCell(stacked_rnn,
                state_is_tuple = True)
        self.states_series_s, self.current_state = tf.nn.dynamic_rnn(cell, self.BX, initial_state=self.rnn_tuple_state)

        self.states_series = tf.reshape(self.states_series_s, [-1, self.settings.state_size], name = 'reshape_state')
        with tf.name_scope('dense'):
            W2 = tf.Variable(np.random.rand(self.settings.state_size, self.settings.output_dim),dtype=tf.float32, name = 'weights')
            b2 = tf.Variable(np.zeros((1,self.settings.output_dim)), dtype=tf.float32, name = 'bias')
            self.logits = tf.matmul(self.states_series, W2) + b2
        
        # Operations
        self.labels = tf.reshape(self.BY, [-1, self.settings.output_dim])
        self.losses = tf.losses.huber_loss(self.labels, self.logits, reduction = tf.losses.Reduction.NONE)
        with tf.name_scope('losses'):
            self.loss = tf.reduce_mean(self.losses)
        with tf.name_scope('accuracy_op'):
            self.accuracy = tf.reduce_mean(tf.square(tf.subtract(self.labels, self.logits)))
        with tf.name_scope('train_step'):
            self.train_step = tf.train.AdamOptimizer(self.settings.learning_rate).minimize(self.loss)
        
        # Tensorboard
        self.tensorboard()


    def rnn_hsX_lX_m1(self):
        # Setup
        self.rnn_placeholders()
        self.rnn_hidden_state()
        # Forward pass
        stacked_rnn = []
        for _ in range(self.settings.num_layers):
            stacked_rnn.append(tf.nn.rnn_cell.BasicRNNCell(self.settings.state_size, activation=tf.nn.tanh, name='rnn_cell_'+str(_)))
        cell = tf.nn.rnn_cell.MultiRNNCell(stacked_rnn, state_is_tuple = True)
        # Feeds a list to the fuck bag
        self.BXR = tf.reshape(self.BX, [-1, self.settings.sequence_length*self.settings.input_dim], name="BX-R")
        self.BYR = tf.reshape(self.BY, [-1, self.settings.sequence_length*self.settings.output_dim], name="BY-R")
        self.inputs_series = tf.split(self.BXR, self.settings.sequence_length, axis=1)
        self.labels_series = tf.split(self.BYR, self.settings.sequence_length, axis=1)
        self.input_one = [self.inputs_series[0]]
        self.input_two = self.inputs_series[1:]
        self.first_series, self.mid_state = tf.nn.static_rnn(cell, self.input_one, initial_state=self.rnn_tuple_state)
        self.second_series, self.current_state = tf.nn.static_rnn(cell, self.input_two, initial_state=self.mid_state)
        self.states_series = self.first_series + self.second_series
        with tf.name_scope('dense'):
            W2 = tf.Variable(np.random.rand(self.settings.state_size, self.settings.output_dim),dtype=tf.float32, name = 'weights')
            b2 = tf.Variable(np.zeros((1,self.settings.output_dim)), dtype=tf.float32, name = 'bias')
            self.logits_series = [tf.matmul(state, W2) + b2 for state in self.states_series]
        self.losses = [tf.losses.huber_loss(logits,labels) for logits, labels in zip(self.logits_series,self.labels_series)]
        self.logits = self.logits_series
        with tf.name_scope('losses'):
            self.loss = tf.reduce_mean(self.losses)
        with tf.name_scope('accuracy_op'):
            self.accuracy = tf.reduce_mean([tf.square(tf.subtract(labels, logits)) for logits, labels in zip(self.logits_series, self.labels_series)])
        with tf.name_scope('train_step'):
            self.train_step = tf.train.AdamOptimizer(self.settings.learning_rate).minimize(self.loss)
        self.tensorboard()

    def rnn_hsX_lX_m2_16(self):
        # Setup
        self.rnn_placeholders()
        self.rnn_hidden_state()
        # Forward pass
        stacked_rnn = []
        for _ in range(self.settings.num_layers):
            stacked_rnn.append(tf.nn.rnn_cell.BasicRNNCell(self.settings.state_size, activation=tf.nn.tanh, name='rnn_cell_'+str(_)))
        cell = tf.nn.rnn_cell.MultiRNNCell(stacked_rnn, state_is_tuple = True)
        # Feeds a list to the fuck bag
        self.BXR = tf.reshape(self.BX, [-1, self.settings.sequence_length*self.settings.input_dim], name="BX-R")
        self.BYR = tf.reshape(self.BY, [-1, self.settings.sequence_length*self.settings.output_dim], name="BY-R")
        self.inputs_series = tf.split(self.BXR, self.settings.sequence_length, axis=1)
        self.labels_series = tf.split(self.BYR, self.settings.sequence_length, axis=1)
        self.input_one = [self.inputs_series[0]]
        self.input_two = self.inputs_series[1:]
        self.first_series, self.mid_state = tf.nn.static_rnn(cell, self.input_one, initial_state=self.rnn_tuple_state)
        self.second_series, self.current_state = tf.nn.static_rnn(cell, self.input_two, initial_state=self.mid_state)
        self.states_series = self.first_series + self.second_series
        with tf.name_scope('dense_1'):
            W2 = tf.Variable(np.random.rand(self.settings.state_size, 16),dtype=tf.float32, name = 'weights')
            b2 = tf.Variable(np.zeros((1,16)), dtype=tf.float32, name = 'bias')
            self.x_series = [tf.matmul(state, W2) + b2 for state in self.states_series]
        with tf.name_scope('dense_2'):
            W3 = tf.Variable(np.random.rand(16, self.settings.output_dim),dtype=tf.float32, name = 'weights')
            b3 = tf.Variable(np.zeros((1,self.settings.output_dim)), dtype=tf.float32, name = 'bias')
            self.logits_series = [tf.matmul(x, W3) + b3 for x in self.x_series]
        self.losses = [tf.losses.huber_loss(logits,labels) for logits, labels in zip(self.logits_series,self.labels_series)]
        self.logits = self.logits_series
        with tf.name_scope('losses'):
            self.loss = tf.reduce_mean(self.losses)
        with tf.name_scope('accuracy_op'):
            self.accuracy = tf.reduce_mean([tf.square(tf.subtract(labels, logits)) for logits, labels in zip(self.logits_series, self.labels_series)])
        with tf.name_scope('train_step'):
            self.train_step = tf.train.AdamOptimizer(self.settings.learning_rate).minimize(self.loss)
        self.tensorboard()

    def rnn_hsX_lX_m2_32(self):
        # Setup
        self.rnn_placeholders()
        self.rnn_hidden_state()
        # Forward pass
        stacked_rnn = []
        for _ in range(self.settings.num_layers):
            stacked_rnn.append(tf.nn.rnn_cell.BasicRNNCell(self.settings.state_size, activation=tf.nn.tanh, name='rnn_cell_'+str(_)))
        cell = tf.nn.rnn_cell.MultiRNNCell(stacked_rnn, state_is_tuple = True)
        # Feeds a list to the fuck bag
        self.BXR = tf.reshape(self.BX, [-1, self.settings.sequence_length*self.settings.input_dim], name="BX-R")
        self.BYR = tf.reshape(self.BY, [-1, self.settings.sequence_length*self.settings.output_dim], name="BY-R")
        self.inputs_series = tf.split(self.BXR, self.settings.sequence_length, axis=1)
        self.labels_series = tf.split(self.BYR, self.settings.sequence_length, axis=1)
        self.input_one = [self.inputs_series[0]]
        self.input_two = self.inputs_series[1:]
        self.first_series, self.mid_state = tf.nn.static_rnn(cell, self.input_one, initial_state=self.rnn_tuple_state)
        self.second_series, self.current_state = tf.nn.static_rnn(cell, self.input_two, initial_state=self.mid_state)
        self.states_series = self.first_series + self.second_series
        with tf.name_scope('dense_1'):
            W2 = tf.Variable(np.random.rand(self.settings.state_size, 32),dtype=tf.float32, name = 'weights')
            b2 = tf.Variable(np.zeros((1,32)), dtype=tf.float32, name = 'bias')
            self.x_series = [tf.matmul(state, W2) + b2 for state in self.states_series]
        with tf.name_scope('dense_2'):
            W3 = tf.Variable(np.random.rand(32, self.settings.output_dim),dtype=tf.float32, name = 'weights')
            b3 = tf.Variable(np.zeros((1,self.settings.output_dim)), dtype=tf.float32, name = 'bias')
            self.logits_series = [tf.matmul(x, W3) + b3 for x in self.x_series]
        self.losses = [tf.losses.huber_loss(logits,labels) for logits, labels in zip(self.logits_series,self.labels_series)]
        self.logits = self.logits_series
        with tf.name_scope('losses'):
            self.loss = tf.reduce_mean(self.losses)
        with tf.name_scope('accuracy_op'):
            self.accuracy = tf.reduce_mean([tf.square(tf.subtract(labels, logits)) for logits, labels in zip(self.logits_series, self.labels_series)])
        with tf.name_scope('train_step'):
            self.train_step = tf.train.AdamOptimizer(self.settings.learning_rate).minimize(self.loss)
        self.tensorboard()

    def rnn_hsX_lX_m2_64(self):
        # Setup
        self.rnn_placeholders()
        self.rnn_hidden_state()
        # Forward pass
        stacked_rnn = []
        for _ in range(self.settings.num_layers):
            stacked_rnn.append(tf.nn.rnn_cell.BasicRNNCell(self.settings.state_size, activation=tf.nn.tanh, name='rnn_cell_'+str(_)))
        cell = tf.nn.rnn_cell.MultiRNNCell(stacked_rnn, state_is_tuple = True)
        # Feeds a list to the fuck bag
        self.BXR = tf.reshape(self.BX, [-1, self.settings.sequence_length*self.settings.input_dim], name="BX-R")
        self.BYR = tf.reshape(self.BY, [-1, self.settings.sequence_length*self.settings.output_dim], name="BY-R")
        self.inputs_series = tf.split(self.BXR, self.settings.sequence_length, axis=1)
        self.labels_series = tf.split(self.BYR, self.settings.sequence_length, axis=1)
        self.input_one = [self.inputs_series[0]]
        self.input_two = self.inputs_series[1:]
        self.first_series, self.mid_state = tf.nn.static_rnn(cell, self.input_one, initial_state=self.rnn_tuple_state)
        self.second_series, self.current_state = tf.nn.static_rnn(cell, self.input_two, initial_state=self.mid_state)
        self.states_series = self.first_series + self.second_series
        with tf.name_scope('dense_1'):
            W2 = tf.Variable(np.random.rand(self.settings.state_size, 64),dtype=tf.float32, name = 'weights')
            b2 = tf.Variable(np.zeros((1,64)), dtype=tf.float32, name = 'bias')
            self.x_series = [tf.matmul(state, W2) + b2 for state in self.states_series]
        with tf.name_scope('dense_2'):
            W3 = tf.Variable(np.random.rand(64, self.settings.output_dim),dtype=tf.float32, name = 'weights')
            b3 = tf.Variable(np.zeros((1,self.settings.output_dim)), dtype=tf.float32, name = 'bias')
            self.logits_series = [tf.matmul(x, W3) + b3 for x in self.x_series]
        self.losses = [tf.losses.huber_loss(logits,labels) for logits, labels in zip(self.logits_series,self.labels_series)]
        self.logits = self.logits_series
        with tf.name_scope('losses'):
            self.loss = tf.reduce_mean(self.losses)
        with tf.name_scope('accuracy_op'):
            self.accuracy = tf.reduce_mean([tf.square(tf.subtract(labels, logits)) for logits, labels in zip(self.logits_series, self.labels_series)])
        with tf.name_scope('train_step'):
            self.train_step = tf.train.AdamOptimizer(self.settings.learning_rate).minimize(self.loss)
        self.tensorboard()
    
    def gru_hsX_lX_m1(self):
        # Setup
        self.rnn_placeholders()
        self.rnn_hidden_state()
        # Forward pass
        stacked_rnn = []
        for _ in range(self.settings.num_layers):
            stacked_rnn.append(tf.nn.rnn_cell.GRUCell(self.settings.state_size, activation=tf.nn.tanh, name='rnn_cell_'+str(_)))
        cell = tf.nn.rnn_cell.MultiRNNCell(stacked_rnn, state_is_tuple = True)
        # Feeds a list to the fuck bag
        self.BXR = tf.reshape(self.BX, [-1, self.settings.sequence_length*self.settings.input_dim], name="BX-R")
        self.BYR = tf.reshape(self.BY, [-1, self.settings.sequence_length*self.settings.output_dim], name="BY-R")
        self.inputs_series = tf.split(self.BXR, self.settings.sequence_length, axis=1)
        self.labels_series = tf.split(self.BYR, self.settings.sequence_length, axis=1)
        self.input_one = [self.inputs_series[0]]
        self.input_two = self.inputs_series[1:]
        self.first_series, self.mid_state = tf.nn.static_rnn(cell, self.input_one, initial_state=self.rnn_tuple_state)
        self.second_series, self.current_state = tf.nn.static_rnn(cell, self.input_two, initial_state=self.mid_state)
        self.states_series = self.first_series + self.second_series
        with tf.name_scope('dense'):
            W2 = tf.Variable(np.random.rand(self.settings.state_size, self.settings.output_dim),dtype=tf.float32, name = 'weights')
            b2 = tf.Variable(np.zeros((1,self.settings.output_dim)), dtype=tf.float32, name = 'bias')
            self.logits_series = [tf.matmul(state, W2) + b2 for state in self.states_series]
        self.losses = [tf.losses.huber_loss(logits,labels) for logits, labels in zip(self.logits_series,self.labels_series)]
        self.logits = self.logits_series
        with tf.name_scope('losses'):
            self.loss = tf.reduce_mean(self.losses)
        with tf.name_scope('accuracy_op'):
            self.accuracy = tf.reduce_mean([tf.square(tf.subtract(labels, logits)) for logits, labels in zip(self.logits_series, self.labels_series)])
        with tf.name_scope('train_step'):
            self.train_step = tf.train.AdamOptimizer(self.settings.learning_rate).minimize(self.loss)
        self.tensorboard()

    def gru_hsX_lX_m2_16(self):
        # Setup
        self.rnn_placeholders()
        self.rnn_hidden_state()
        # Forward pass
        stacked_rnn = []
        for _ in range(self.settings.num_layers):
            stacked_rnn.append(tf.nn.rnn_cell.GRUCell(self.settings.state_size, activation=tf.nn.tanh, name='rnn_cell_'+str(_)))
        cell = tf.nn.rnn_cell.MultiRNNCell(stacked_rnn, state_is_tuple = True)
        # Feeds a list to the fuck bag
        self.BXR = tf.reshape(self.BX, [-1, self.settings.sequence_length*self.settings.input_dim], name="BX-R")
        self.BYR = tf.reshape(self.BY, [-1, self.settings.sequence_length*self.settings.output_dim], name="BY-R")
        self.inputs_series = tf.split(self.BXR, self.settings.sequence_length, axis=1)
        self.labels_series = tf.split(self.BYR, self.settings.sequence_length, axis=1)
        self.input_one = [self.inputs_series[0]]
        self.input_two = self.inputs_series[1:]
        self.first_series, self.mid_state = tf.nn.static_rnn(cell, self.input_one, initial_state=self.rnn_tuple_state)
        self.second_series, self.current_state = tf.nn.static_rnn(cell, self.input_two, initial_state=self.mid_state)
        self.states_series = self.first_series + self.second_series
        with tf.name_scope('dense_1'):
            W2 = tf.Variable(np.random.rand(self.settings.state_size, 16),dtype=tf.float32, name = 'weights')
            b2 = tf.Variable(np.zeros((1,16)), dtype=tf.float32, name = 'bias')
            self.x_series = [tf.matmul(state, W2) + b2 for state in self.states_series]
        with tf.name_scope('dense_2'):
            W3 = tf.Variable(np.random.rand(16, self.settings.output_dim),dtype=tf.float32, name = 'weights')
            b3 = tf.Variable(np.zeros((1,self.settings.output_dim)), dtype=tf.float32, name = 'bias')
            self.logits_series = [tf.matmul(x, W3) + b3 for x in self.x_series]
        self.losses = [tf.losses.huber_loss(logits,labels) for logits, labels in zip(self.logits_series,self.labels_series)]
        self.logits = self.logits_series
        with tf.name_scope('losses'):
            self.loss = tf.reduce_mean(self.losses)
        with tf.name_scope('accuracy_op'):
            self.accuracy = tf.reduce_mean([tf.square(tf.subtract(labels, logits)) for logits, labels in zip(self.logits_series, self.labels_series)])
        with tf.name_scope('train_step'):
            self.train_step = tf.train.AdamOptimizer(self.settings.learning_rate).minimize(self.loss)
        self.tensorboard()

    def gru_hsX_lX_m2_32(self):
        # Setup
        self.rnn_placeholders()
        self.rnn_hidden_state()
        # Forward pass
        stacked_rnn = []
        for _ in range(self.settings.num_layers):
            stacked_rnn.append(tf.nn.rnn_cell.GRUCell(self.settings.state_size, activation=tf.nn.tanh, name='rnn_cell_'+str(_)))
        cell = tf.nn.rnn_cell.MultiRNNCell(stacked_rnn, state_is_tuple = True)
        # Feeds a list to the fuck bag
        self.BXR = tf.reshape(self.BX, [-1, self.settings.sequence_length*self.settings.input_dim], name="BX-R")
        self.BYR = tf.reshape(self.BY, [-1, self.settings.sequence_length*self.settings.output_dim], name="BY-R")
        self.inputs_series = tf.split(self.BXR, self.settings.sequence_length, axis=1)
        self.labels_series = tf.split(self.BYR, self.settings.sequence_length, axis=1)
        self.input_one = [self.inputs_series[0]]
        self.input_two = self.inputs_series[1:]
        self.first_series, self.mid_state = tf.nn.static_rnn(cell, self.input_one, initial_state=self.rnn_tuple_state)
        self.second_series, self.current_state = tf.nn.static_rnn(cell, self.input_two, initial_state=self.mid_state)
        self.states_series = self.first_series + self.second_series
        with tf.name_scope('dense_1'):
            W2 = tf.Variable(np.random.rand(self.settings.state_size, 32),dtype=tf.float32, name = 'weights')
            b2 = tf.Variable(np.zeros((1,32)), dtype=tf.float32, name = 'bias')
            self.x_series = [tf.matmul(state, W2) + b2 for state in self.states_series]
        with tf.name_scope('dense_2'):
            W3 = tf.Variable(np.random.rand(32, self.settings.output_dim),dtype=tf.float32, name = 'weights')
            b3 = tf.Variable(np.zeros((1,self.settings.output_dim)), dtype=tf.float32, name = 'bias')
            self.logits_series = [tf.matmul(x, W3) + b3 for x in self.x_series]
        self.losses = [tf.losses.huber_loss(logits,labels) for logits, labels in zip(self.logits_series,self.labels_series)]
        self.logits = self.logits_series
        with tf.name_scope('losses'):
            self.loss = tf.reduce_mean(self.losses)
        with tf.name_scope('accuracy_op'):
            self.accuracy = tf.reduce_mean([tf.square(tf.subtract(labels, logits)) for logits, labels in zip(self.logits_series, self.labels_series)])
        with tf.name_scope('train_step'):
            self.train_step = tf.train.AdamOptimizer(self.settings.learning_rate).minimize(self.loss)
        self.tensorboard()

    def gru_hsX_lX_m2_64(self):
        # Setup
        self.rnn_placeholders()
        self.rnn_hidden_state()
        # Forward pass
        stacked_rnn = []
        for _ in range(self.settings.num_layers):
            stacked_rnn.append(tf.nn.rnn_cell.GRUCell(self.settings.state_size, activation=tf.nn.tanh, name='rnn_cell_'+str(_)))
        cell = tf.nn.rnn_cell.MultiRNNCell(stacked_rnn, state_is_tuple = True)
        # Feeds a list to the fuck bag
        self.BXR = tf.reshape(self.BX, [-1, self.settings.sequence_length*self.settings.input_dim], name="BX-R")
        self.BYR = tf.reshape(self.BY, [-1, self.settings.sequence_length*self.settings.output_dim], name="BY-R")
        self.inputs_series = tf.split(self.BXR, self.settings.sequence_length, axis=1)
        self.labels_series = tf.split(self.BYR, self.settings.sequence_length, axis=1)
        self.input_one = [self.inputs_series[0]]
        self.input_two = self.inputs_series[1:]
        self.first_series, self.mid_state = tf.nn.static_rnn(cell, self.input_one, initial_state=self.rnn_tuple_state)
        self.second_series, self.current_state = tf.nn.static_rnn(cell, self.input_two, initial_state=self.mid_state)
        self.states_series = self.first_series + self.second_series
        with tf.name_scope('dense_1'):
            W2 = tf.Variable(np.random.rand(self.settings.state_size, 64),dtype=tf.float32, name = 'weights')
            b2 = tf.Variable(np.zeros((1,64)), dtype=tf.float32, name = 'bias')
            self.x_series = [tf.matmul(state, W2) + b2 for state in self.states_series]
        with tf.name_scope('dense_2'):
            W3 = tf.Variable(np.random.rand(64, self.settings.output_dim),dtype=tf.float32, name = 'weights')
            b3 = tf.Variable(np.zeros((1,self.settings.output_dim)), dtype=tf.float32, name = 'bias')
            self.logits_series = [tf.matmul(x, W3) + b3 for x in self.x_series]
        self.losses = [tf.losses.huber_loss(logits,labels) for logits, labels in zip(self.logits_series,self.labels_series)]
        self.logits = self.logits_series
        with tf.name_scope('losses'):
            self.loss = tf.reduce_mean(self.losses)
        with tf.name_scope('accuracy_op'):
            self.accuracy = tf.reduce_mean([tf.square(tf.subtract(labels, logits)) for logits, labels in zip(self.logits_series, self.labels_series)])
        with tf.name_scope('train_step'):
            self.train_step = tf.train.AdamOptimizer(self.settings.learning_rate).minimize(self.loss)
        self.tensorboard()
    
    def lstm_hsX_lX_m1(self):
        # Setup
        self.lstm_placeholders()
        self.lstm_hidden_state()
        # Forward pass
        stacked_rnn = []
        for _ in range(self.settings.num_layers):
            stacked_rnn.append(tf.nn.rnn_cell.LSTMCell(self.settings.state_size, state_is_tuple=True, name='lstm_cell_'+str(_)))
        cell = tf.nn.rnn_cell.MultiRNNCell(stacked_rnn, state_is_tuple = True)
        # Feeds a list to the fuck bag
        self.BXR = tf.reshape(self.BX, [-1, self.settings.sequence_length*self.settings.input_dim], name="BX-R")
        self.BYR = tf.reshape(self.BY, [-1, self.settings.sequence_length*self.settings.output_dim], name="BY-R")
        self.inputs_series = tf.split(self.BXR, self.settings.sequence_length, axis=1)
        self.labels_series = tf.split(self.BYR, self.settings.sequence_length, axis=1)
        self.input_one = [self.inputs_series[0]]
        self.input_two = self.inputs_series[1:]
        self.first_series, self.mid_state = tf.nn.static_rnn(cell, self.input_one, initial_state=self.rnn_tuple_state)
        self.second_series, self.current_state = tf.nn.static_rnn(cell, self.input_two, initial_state=self.mid_state)
        self.states_series = self.first_series + self.second_series
        with tf.name_scope('dense'):
            W2 = tf.Variable(np.random.rand(self.settings.state_size, self.settings.output_dim),dtype=tf.float32, name = 'weights')
            b2 = tf.Variable(np.zeros((1,self.settings.output_dim)), dtype=tf.float32, name = 'bias')
            self.logits_series = [tf.matmul(state, W2) + b2 for state in self.states_series]
        self.losses = [tf.losses.huber_loss(logits,labels) for logits, labels in zip(self.logits_series,self.labels_series)]
        self.logits = self.logits_series
        with tf.name_scope('losses'):
            self.loss = tf.reduce_mean(self.losses)
        with tf.name_scope('accuracy_op'):
            self.accuracy = tf.reduce_mean([tf.square(tf.subtract(labels, logits)) for logits, labels in zip(self.logits_series, self.labels_series)])
        with tf.name_scope('train_step'):
            self.train_step = tf.train.AdamOptimizer(self.settings.learning_rate).minimize(self.loss)
        self.tensorboard()

    def lstm_hsX_lX_m2_l16(self):
        # Setup
        self.lstm_placeholders()
        self.lstm_hidden_state()
        # Forward pass
        stacked_rnn = []
        for _ in range(self.settings.num_layers):
            stacked_rnn.append(tf.nn.rnn_cell.LSTMCell(self.settings.state_size, state_is_tuple=True, name='lstm_cell_'+str(_)))
        cell = tf.nn.rnn_cell.MultiRNNCell(stacked_rnn, state_is_tuple = True)
        # Feeds a list to the fuck bag
        self.BXR = tf.reshape(self.BX, [-1, self.settings.sequence_length*self.settings.input_dim], name="BX-R")
        self.BYR = tf.reshape(self.BY, [-1, self.settings.sequence_length*self.settings.output_dim], name="BY-R")
        self.inputs_series = tf.split(self.BXR, self.settings.sequence_length, axis=1)
        self.labels_series = tf.split(self.BYR, self.settings.sequence_length, axis=1)
        self.input_one = [self.inputs_series[0]]
        self.input_two = self.inputs_series[1:]
        self.first_series, self.mid_state = tf.nn.static_rnn(cell, self.input_one, initial_state=self.rnn_tuple_state)
        self.second_series, self.current_state = tf.nn.static_rnn(cell, self.input_two, initial_state=self.mid_state)
        self.states_series = self.first_series + self.second_series
        with tf.name_scope('dense_1'):
            W2 = tf.Variable(np.random.rand(self.settings.state_size, 16),dtype=tf.float32, name = 'weights')
            b2 = tf.Variable(np.zeros((1,16)), dtype=tf.float32, name = 'bias')
            self.x_series = [tf.matmul(state, W2) + b2 for state in self.states_series]
        with tf.name_scope('dense_2'):
            W3 = tf.Variable(np.random.rand(16, self.settings.output_dim),dtype=tf.float32, name = 'weights')
            b3 = tf.Variable(np.zeros((1,self.settings.output_dim)), dtype=tf.float32, name = 'bias')
            self.logits_series = [tf.matmul(x, W3) + b3 for x in self.x_series]

        self.losses = [tf.losses.huber_loss(logits,labels) for logits, labels in zip(self.logits_series,self.labels_series)]
        self.logits = self.logits_series
        with tf.name_scope('losses'):
            self.loss = tf.reduce_mean(self.losses)
        with tf.name_scope('accuracy_op'):
            self.accuracy = tf.reduce_mean([tf.square(tf.subtract(labels, logits)) for logits, labels in zip(self.logits_series, self.labels_series)])
        with tf.name_scope('train_step'):
            self.train_step = tf.train.AdamOptimizer(self.settings.learning_rate).minimize(self.loss)
        self.tensorboard()
    
    def lstm_hsX_lX_m2_l32(self):
        # Setup
        self.lstm_placeholders()
        self.lstm_hidden_state()
        # Forward pass
        stacked_rnn = []
        for _ in range(self.settings.num_layers):
            stacked_rnn.append(tf.nn.rnn_cell.LSTMCell(self.settings.state_size, state_is_tuple=True, name='lstm_cell_'+str(_)))
        cell = tf.nn.rnn_cell.MultiRNNCell(stacked_rnn, state_is_tuple = True)
        # Feeds a list to the fuck bag
        self.BXR = tf.reshape(self.BX, [-1, self.settings.sequence_length*self.settings.input_dim], name="BX-R")
        self.BYR = tf.reshape(self.BY, [-1, self.settings.sequence_length*self.settings.output_dim], name="BY-R")
        self.inputs_series = tf.split(self.BXR, self.settings.sequence_length, axis=1)
        self.labels_series = tf.split(self.BYR, self.settings.sequence_length, axis=1)
        self.input_one = [self.inputs_series[0]]
        self.input_two = self.inputs_series[1:]
        self.first_series, self.mid_state = tf.nn.static_rnn(cell, self.input_one, initial_state=self.rnn_tuple_state)
        self.second_series, self.current_state = tf.nn.static_rnn(cell, self.input_two, initial_state=self.mid_state)
        self.states_series = self.first_series + self.second_series
        with tf.name_scope('dense_1'):
            W2 = tf.Variable(np.random.rand(self.settings.state_size, 32),dtype=tf.float32, name = 'weights')
            b2 = tf.Variable(np.zeros((1,32)), dtype=tf.float32, name = 'bias')
            self.x_series = [tf.matmul(state, W2) + b2 for state in self.states_series]
        with tf.name_scope('dense_2'):
            W3 = tf.Variable(np.random.rand(32, self.settings.output_dim),dtype=tf.float32, name = 'weights')
            b3 = tf.Variable(np.zeros((1,self.settings.output_dim)), dtype=tf.float32, name = 'bias')
            self.logits_series = [tf.matmul(x, W3) + b3 for x in self.x_series]

        self.losses = [tf.losses.huber_loss(logits,labels) for logits, labels in zip(self.logits_series,self.labels_series)]
        self.logits = self.logits_series
        with tf.name_scope('losses'):
            self.loss = tf.reduce_mean(self.losses)
        with tf.name_scope('accuracy_op'):
            self.accuracy = tf.reduce_mean([tf.square(tf.subtract(labels, logits)) for logits, labels in zip(self.logits_series, self.labels_series)])
        with tf.name_scope('train_step'):
            self.train_step = tf.train.AdamOptimizer(self.settings.learning_rate).minimize(self.loss)
        self.tensorboard()
    
    def lstm_hsX_lX_m2_l64(self):
        # Setup
        self.lstm_placeholders()
        self.lstm_hidden_state()
        # Forward pass
        stacked_rnn = []
        for _ in range(self.settings.num_layers):
            stacked_rnn.append(tf.nn.rnn_cell.LSTMCell(self.settings.state_size, state_is_tuple=True, name='lstm_cell_'+str(_)))
        cell = tf.nn.rnn_cell.MultiRNNCell(stacked_rnn, state_is_tuple = True)
        # Feeds a list to the fuck bag
        self.BXR = tf.reshape(self.BX, [-1, self.settings.sequence_length*self.settings.input_dim], name="BX-R")
        self.BYR = tf.reshape(self.BY, [-1, self.settings.sequence_length*self.settings.output_dim], name="BY-R")
        self.inputs_series = tf.split(self.BXR, self.settings.sequence_length, axis=1)
        self.labels_series = tf.split(self.BYR, self.settings.sequence_length, axis=1)
        self.input_one = [self.inputs_series[0]]
        self.input_two = self.inputs_series[1:]
        self.first_series, self.mid_state = tf.nn.static_rnn(cell, self.input_one, initial_state=self.rnn_tuple_state)
        self.second_series, self.current_state = tf.nn.static_rnn(cell, self.input_two, initial_state=self.mid_state)
        self.states_series = self.first_series + self.second_series
        with tf.name_scope('dense_1'):
            W2 = tf.Variable(np.random.rand(self.settings.state_size, 64),dtype=tf.float32, name = 'weights')
            b2 = tf.Variable(np.zeros((1,64)), dtype=tf.float32, name = 'bias')
            self.x_series = [tf.matmul(state, W2) + b2 for state in self.states_series]
        with tf.name_scope('dense_2'):
            W3 = tf.Variable(np.random.rand(64, self.settings.output_dim),dtype=tf.float32, name = 'weights')
            b3 = tf.Variable(np.zeros((1,self.settings.output_dim)), dtype=tf.float32, name = 'bias')
            self.logits_series = [tf.matmul(x, W3) + b3 for x in self.x_series]

        self.losses = [tf.losses.huber_loss(logits,labels) for logits, labels in zip(self.logits_series,self.labels_series)]
        self.logits = self.logits_series
        with tf.name_scope('losses'):
            self.loss = tf.reduce_mean(self.losses)
        with tf.name_scope('accuracy_op'):
            self.accuracy = tf.reduce_mean([tf.square(tf.subtract(labels, logits)) for logits, labels in zip(self.logits_series, self.labels_series)])
        with tf.name_scope('train_step'):
            self.train_step = tf.train.AdamOptimizer(self.settings.learning_rate).minimize(self.loss)
        self.tensorboard()
