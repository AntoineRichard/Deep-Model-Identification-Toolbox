import numpy as np
from settings import Settings
from dataset import DataPreProcessor
from neuralnet import DynamicLSTM

def custom_fn(loss):
    raise Exception('Not implemented')

class RandomSampler:
    def __init__(self, data, settings, nn):
        self.data = data
        self.settings = settings
        self.nn = nn
        self.x_train, self.y_train, self.x_test, self.y_test, self.x_val, self.y_val = self.data.generate_dynamic_data()
        self.val_idx = 0
        self.test_idx = 0
        self.train_idx = 0
    
    def next_test_batch(self, _current_state):
        if (self.test_idx == 0):
            _current_state = self.nn.reset_state

        #start_idx = self.test_idx * self.settings.sequence_length
        #end_idx = start_idx + self.settings.sequence_length
        start_idx = int(np.random.rand(1)[0]*self.settings.num_batches_train*self.settings.sequence_length)
        end_idx = start_idx + self.settings.sequence_length

        batchX = self.x_test[:,start_idx:end_idx,:]
        batchY = self.y_test[:,start_idx:end_idx,:]

        if self.test_idx == self.settings.num_batches_test:
            self.test_idx = 0
            return 0, batchX, batchY, _current_state
        else:
            self.test_idx += 1
            return 1, batchX, batchY, _current_state
        
    def next_val_batch(self, _current_state):
        if (self.val_idx == 0):
            _current_state = self.nn.reset_state

        start_idx = self.val_idx * self.settings.sequence_length
        end_idx = start_idx + self.settings.sequence_length

        batchX = self.x_val[:,start_idx:end_idx,:]
        batchY = self.y_val[:,start_idx:end_idx,:]
        
        if self.val_idx == self.settings.num_batches_val:
            self.val_idx = 0
            return 0, batchX, batchY, _current_state
        else:
            self.val_idx += 1
            return 1, batchX, batchY, _current_state
        
    def next_train_batch(self, _current_state):
        if (self.train_idx == 0):
            _current_state = self.nn.reset_state

        start_idx = self.train_idx * self.settings.sequence_length
        end_idx = start_idx + self.settings.sequence_length
        
        batchX = self.x_train[:,start_idx:end_idx,:]
        batchY = self.y_train[:,start_idx:end_idx,:]

        if self.train_idx%self.settings.log_frequency == 0:
            log = True
        else:
            log = False

        if self.train_idx == self.settings.num_batches_train:
            self.train_idx = 0
            return 0, log, batchX, batchY, _current_state, None
        else:
            self.train_idx += 1
            return 1, log, batchX, batchY, _current_state, None

class Sampler:
    def __init__(self, data, settings, nn):
        self.data = data
        self.settings = settings
        self.nn = nn
        self.x_train, self.y_train, self.x_test, self.y_test, self.x_val, self.y_val = self.data.generate_dynamic_data()
        self.val_idx = 0
        self.test_idx = 0
        self.train_idx = 0
    
    def next_test_batch(self, _current_state):
        if (self.test_idx == 0):
            _current_state = self.nn.reset_state

        start_idx = self.test_idx * self.settings.sequence_length
        end_idx = start_idx + self.settings.sequence_length

        batchX = self.x_test[:,start_idx:end_idx,:]
        batchY = self.y_test[:,start_idx:end_idx,:]

        if self.test_idx == self.settings.num_batches_test:
            self.test_idx = 0
            return 0, batchX, batchY, _current_state
        else:
            self.test_idx += 1
            return 1, batchX, batchY, _current_state
        
        
    def next_val_batch(self, _current_state):
        if (self.val_idx == 0):
            _current_state = self.nn.reset_state

        start_idx = self.val_idx * self.settings.sequence_length
        end_idx = start_idx + self.settings.sequence_length

        batchX = self.x_val[:,start_idx:end_idx,:]
        batchY = self.y_val[:,start_idx:end_idx,:]
        
        if self.val_idx == self.settings.num_batches_val:
            self.val_idx = 0
            return 0, batchX, batchY, _current_state
        else:
            self.val_idx += 1
            return 1, batchX, batchY, _current_state
        
    def next_train_batch(self, _current_state):
        if (self.train_idx == 0):
            _current_state = self.nn.reset_state
            self.x_train, self.y_train = self.data.shift_train()

        start_idx = self.train_idx * self.settings.sequence_length
        end_idx = start_idx + self.settings.sequence_length
        
        batchX = self.x_train[:,start_idx:end_idx,:]
        batchY = self.y_train[:,start_idx:end_idx,:]

        if self.train_idx%self.settings.log_frequency == 0:
            log = True
        else:
            log = False

        if self.train_idx == self.settings.num_batches_train:
            self.train_idx = 0
            return 0, log, batchX, batchY, _current_state, None
        else:
            self.train_idx += 1
            return 1, log, batchX, batchY, _current_state, None
    
    def next_val_trajectory(self, count):
        # Fixed trajectory for the whole training: it would be better to pick
        # the trajectories manually and focus on hard samples. Adds warm up
        # trajectories
        #step = int((self.x_val.shape[1] - self.settings.window_size)/self.settings.max_trajectories)
        #trajectory_table = list(range(0,self.x_val.shape[1]-self.settings.window_size,step))
        trajectory_table = list(range(0,2000,20))
        _current_state = self.nn.reset_state

        start_idx = trajectory_table[count]
        end_idx = trajectory_table[count] + self.settings.sequence_length*3 + self.settings.window_size
        batchX = self.x_val.copy()[:,start_idx:end_idx,:]
        batchX[:,:,:] = self.x_val.copy()[0,start_idx:end_idx,:]
        batchY = self.y_val.copy()[:,start_idx:end_idx,:]
        batchY[:,:,:] = self.y_val.copy()[0,start_idx:end_idx,:]
        return batchX, batchY, _current_state


class ImportanceSampler:
    def __init__(self, data, settings, nn):
        self.data = data
        self.settings = settings
        self.nn = nn
        self.x_train, self.y_train, self.x_test, self.y_test, self.x_val, self.y_val = self.data.generate_dynamic_data()
        self.val_idx = 0
        self.test_idx = 0
        self.train_idx = 0
        self.loss_array = np.ones([self.settings.num_batches_train+1, self.settings.batch_size*self.settings.sequence_length])
        self.weights = np.ones([self.settings.num_batches_train+1, self.settings.batch_size*self.settings.sequence_length])
    
    def next_test_batch(self, _current_state):
        if (self.test_idx == 0):
            _current_state = self.nn.reset_state

        start_idx = self.test_idx * self.settings.sequence_length
        end_idx = start_idx + self.settings.sequence_length

        batchX = self.x_test[:,start_idx:end_idx,:]
        batchY = self.y_test[:,start_idx:end_idx,:]

        if self.test_idx == self.settings.num_batches_test:
            self.test_idx = 0
            return 0, batchX, batchY, _current_state
        else:
            self.test_idx += 1
            return 1, batchX, batchY, _current_state
        
    def next_val_batch(self, _current_state):
        if (self.val_idx == 0):
            _current_state = self.nn.reset_state

        start_idx = self.val_idx * self.settings.sequence_length
        end_idx = start_idx + self.settings.sequence_length

        batchX = self.x_val[:,start_idx:end_idx,:]
        batchY = self.y_val[:,start_idx:end_idx,:]
        
        if self.val_idx == self.settings.num_batches_val:
            self.val_idx = 0
            return 0, batchX, batchY, _current_state
        else:
            self.val_idx += 1
            return 1, batchX, batchY, _current_state

    def next_train_batch_no_weights(self, _current_state):
        if (self.train_idx == 0):
            _current_state = self.nn.reset_state

        start_idx = self.train_idx * self.settings.sequence_length
        end_idx = start_idx + self.settings.sequence_length
        
        batchX = self.x_train[:,start_idx:end_idx,:]
        batchY = self.y_train[:,start_idx:end_idx,:]

        if self.train_idx%self.settings.log_frequency == 0:
            log = True
        else:
            log = False

        if self.train_idx == self.settings.num_batches_train:
            self.train_idx = 0
            return 0, log, batchX, batchY, _current_state
        else:
            self.train_idx += 1
            return 1, log, batchX, batchY, _current_state

    def next_train_batch(self, _current_state):
        if (self.train_idx == 0):
            _current_state = self.nn.reset_state

        start_idx = self.train_idx * self.settings.sequence_length
        end_idx = start_idx + self.settings.sequence_length
        
        batchX = self.x_train[:,start_idx:end_idx,:]
        batchY = self.y_train[:,start_idx:end_idx,:]

        if self.train_idx%self.settings.log_frequency == 0:
            log = True
        else:
            log = False

        if self.train_idx == self.settings.num_batches_train:
            prev = self.train_idx
            self.train_idx = 0
            return 0, log, batchX, batchY, _current_state, self.probs[prev,:]
        else:
            prev = self.train_idx
            self.train_idx += 1
            return 1, log, batchX, batchY, _current_state, self.probs[prev,:]

    def update_weights(self, sess):
        _current_state = self.nn.reset_state
        tmp_train = self.train_idx
        self.train_idx = 0
        for batch_idx in range(self.settings.num_batches_train+1):
            _, _, batchX, batchY, _current_state = self.next_train_batch_no_weights(_current_state)
            _total_loss, _current_state = sess.run([self.nn.losses, self.nn.current_state],
                feed_dict={self.nn.BX: batchX,
                           self.nn.BY: batchY,
                           self.nn.HS: _current_state})
            self.loss_array[batch_idx, :] = np.mean(_total_loss, axis = 1)
        self.weights, self.probs = self.compute_weights()
        self.train_idx = tmp_train

    def compute_weights(self):
        if self.settings.weighting_mode == 'PER':
            return self.per_weighting(self.loss_array)
        elif self.settings.weighting_mode == 'basic':
            return self.basic_weighting(self.loss_array)
        elif self.settings.weighting_mode == 'custom':
            return self.fn_weighting(self.loss_array, custom_fn)
        else:
            raise Exception('Unknown weighting mode. Available modes are PER, basic and custom')
        
    def per_weighting(self, loss):
        error = loss + self.settings.epsilon
        values = np.power(error, self.settings.alpha)
        num_samples = self.settings.num_batches_train * self.settings.batch_size * self.settings.sequence_length
        probs = num_samples * values/(np.sum(values))
        weights = np.power(probs, - self.settings.beta)
        return weights, probs*weights

    def basic_weighting(self, loss):
        error = loss + self.settings.epsilon
        weights = np.mean(error)
        return weights

    def fn_weighting(self, loss, fn):
        error = loss + self.settings.epsilon
        return fn(error)
