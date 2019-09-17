import numpy as np
import os
import tensorflow as tf
import time

from progress.bar import Bar
from sklearn.metrics import mean_squared_error as SK_MSE

from settings import Settings
from dataset import DataPreProcessor
from neuralnet import DynamicLSTM
from sampler import Sampler
from sampler import ImportanceSampler
from sampler import RandomSampler

class Train:
    def __init__(self):
        # Instantiate objetcs
        self.settings = Settings()
        if os.path.exists(os.path.join(self.settings.output_dir,"test_multi_step_loss_log.npy")):
            exit(0)
        self.data = DataPreProcessor(self.settings)
        self.nn = DynamicLSTM(self.settings)
        self.bar = None

        # Generates dataset
        self.settings.data_parameters(self.data)
        if self.settings.priorization == 'PER':
            self.sampler = ImportanceSampler(self.data, self.settings, self.nn)
        else:
            self.sampler = Sampler(self.data, self.settings, self.nn)

        # Quick-grab
        self.sess = None
        self.val_time_history = []
        self.val_step_history = []
        self.val_loss_history = []
        self.val_acc_history = []
        self.train_time_history = []
        self.train_step_history = []
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_ms_time_history = []
        self.val_ms_step_history = []
        self.val_ms_acc_history  = []

    def train_op(self, batchX, batchY, _current_state, weights = None):
        if weights is None:
            _total_loss, _train_step, _current_state, _predictions_series = self.sess.run(
                [self.nn.loss, self.nn.train_step, self.nn.current_state, self.nn.logits],
                #[self.nn.loss, self.nn.train_step, self.nn.mid_state, self.nn.logits],
                feed_dict={
                    self.nn.BX: batchX,
                    self.nn.BY: batchY,
                    self.nn.HS: _current_state
                })
        else:
            _total_loss, _train_step, _current_state, _predictions_series = self.sess.run(
                [self.nn.loss, self.nn.train_step, self.nn.current_state, self.nn.logits],
                #[self.nn.loss, self.nn.train_step, self.nn.mid_state, self.nn.logits],
                feed_dict={
                    self.nn.BX: batchX,
                    self.nn.BY: batchY,
                    self.nn.HS: _current_state,
                    self.nn.W: weights
                })
        return _total_loss, _train_step, _current_state, _predictions_series

    def train_log_op(self, batchX, batchY, _current_state, weights = None):
        if weights is None:
            _total_loss, _train_step, _current_state, _predictions_series, _summaries, _accuracy = self.sess.run(
                [self.nn.loss, self.nn.train_step, self.nn.current_state, self.nn.logits,
                #[self.nn.loss, self.nn.train_step, self.nn.mid_state, self.nn.logits,
                    self.nn.summaries, self.nn.accuracy],
                feed_dict={
                    self.nn.BX: batchX,
                    self.nn.BY: batchY,
                    self.nn.HS: _current_state
                })
        else:
            _total_loss, _train_step, _current_state, _predictions_series, _summaries, _accuracy = self.sess.run(
                [self.nn.loss, self.nn.train_step, self.nn.current_state, self.nn.logits,
                #[self.nn.loss, self.nn.train_step, self.nn.mid_state, self.nn.logits,
                    self.nn.summaries, self.nn.accuracy],
                feed_dict={
                    self.nn.BX: batchX,
                    self.nn.BY: batchY,
                    self.nn.HS: _current_state,
                    self.nn.W: weights
                })
        return _total_loss, _train_step, _current_state, _predictions_series, _summaries, _accuracy

    def log_train(self, _summaries, loss, acc):
        # CONSOLE
        self.bar.next()
        # TENSORBOARD
        self.train_writer.add_summary(_summaries, self.epoch_idx * self.settings.num_batches_train + self.sampler.train_idx)
        # HARD-LOG
        self.train_time_history.append(time.time())
        self.train_step_history.append(self.epoch_idx * self.settings.num_batches_train + self.sampler.train_idx)
        self.train_loss_history.append(loss)
        self.train_acc_history.append(acc)

    def test_op(self, batchX, batchY, _current_state):
        _total_loss, _current_state, _predictions_series, _accuracy = self.sess.run(
             #[self.nn.loss, self.nn.mid_state, self.nn.logits, self.nn.accuracy],
             [self.nn.loss, self.nn.current_state, self.nn.logits, self.nn.accuracy],
             feed_dict={
                 self.nn.BX: batchX,
                 self.nn.BY: batchY,
                 self.nn.HS: _current_state
             })
        return _total_loss, _current_state, _predictions_series, _accuracy
    
    def traj_op(self, batchX, _current_state):
        _mid_state, _predictions_series  = self.sess.run(
             [self.nn.mid_state, self.nn.logits],
             feed_dict={
                 self.nn.BX: batchX,
                 self.nn.HS: _current_state
             })
        return _mid_state, _predictions_series

    def log_val(self, loss_val, acc_val):
        mean_loss = np.mean(loss_val[2:])
        mean_acc = np.mean(acc_val[2:])
        # CONSOLE
        print('Validation single step')
        print('Loss        : ', mean_loss)
        print('Accuracy    : ', mean_acc)
        # TENSORBOARD
        summary = tf.Summary()
        summary.value.add(tag='loss', simple_value = mean_loss)
        summary.value.add(tag='accuracy', simple_value = mean_acc)
        self.test_writer.add_summary(summary, self.epoch_idx * self.settings.num_batches_train)
        # HARD-LOG
        self.val_time_history.append(time.time())
        self.val_step_history.append(self.epoch_idx * self.settings.num_batches_train)
        self.val_loss_history.append(mean_loss)
        self.val_acc_history.append(mean_acc)
        # SAVE
        if mean_acc < self.best:
            self.best = mean_acc
            self.save('best_1S_NN')
    
    def log_val_ms(self, acc_val):
        # CONSOLE
        print('Validation multi step')
        print('Accuracy    : ', acc_val)
        # HARD-LOG
        self.val_ms_time_history.append(time.time())
        self.val_ms_step_history.append(self.epoch_idx * self.settings.num_batches_train)
        self.val_ms_acc_history.append(acc_val)
        # SAVE
        if acc_val < self.best_ms:
            self.best_ms = acc_val
            self.save('best_NN')

    def train(self):
        self.best = 10000
        self.best_ms = 10000
        with tf.Session() as self.sess:
            # TENSORBOARD
            if self.settings.allow_tb:
                self.train_writer = tf.summary.FileWriter(self.settings.tb_dir +'/'+self.settings.tb_log_name+'/train', self.sess.graph)
                self.test_writer = tf.summary.FileWriter(self.settings.tb_dir +'/'+self.settings.tb_log_name+'/test')
        
            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()

        
            for self.epoch_idx in range(self.settings.num_epochs):
                # RUN VALIDATION
                not_done = 1
                loss_val = []
                acc_val = []
                _current_state = self.nn.reset_state
                while(not_done):
                    not_done, batchX, batchY, _current_state = self.sampler.next_val_batch(_current_state)
                    _total_loss, _current_state, _predictions_series, _accuracy = self.test_op(batchX, batchY, _current_state)
                    loss_val.append(_total_loss)
                    acc_val.append(_accuracy)
                self.log_val(loss_val, acc_val)
                # RUN TRAJECTORY VALIDATION
                avg = [] 
                for j in range(self.settings.max_trajectories):
                    batch_x, batch_y, _current_state = self.sampler.next_val_trajectory(j)
                    # WARM-UP THE LSTM FIRST
                    loss, _current_state, pred, acc = self.test_op(batch_x[:,:self.settings.sequence_length,:], batch_y[:,:self.settings.sequence_length,:], _current_state)
                    loss, _mid_state, pred, acc = self.test_op(batch_x[:,self.settings.sequence_length:self.settings.sequence_length*2,:], batch_y[:,self.settings.sequence_length:self.settings.sequence_length*2], _current_state)
                    # DO THE ACTUAL TRAJECTORY
                    full = batch_x[:,self.settings.sequence_length*2:self.settings.sequence_length*3,:]
                    predictions = []
                    for k in range(self.settings.window_size - 1):
                        _mid_state, pred = self.traj_op(full, _mid_state)
                        pred = pred[0][-1]
                        predictions.append(pred.tolist())
                        pred = pred[np.newaxis,...]
                        cmd = batch_x[0,self.settings.sequence_length*3+k+1, -self.settings.cmd_dim:][np.newaxis,...]
                        new = np.concatenate((pred, cmd), axis=1)[np.newaxis,...]
                        old = full[0,1:,:][np.newaxis,...]
                        full[:,:,:] = np.concatenate((new,old), axis=1)
                    predictions = np.array(predictions)
                    error = SK_MSE(predictions[:,:],batch_y[0,self.settings.sequence_length*3:-1,:])
                    avg.append(error)
                err = np.mean(avg,axis=0)
                avg = np.mean(avg)
                self.log_val_ms(avg)

                # RUN TRAIN
                not_done = 1
                if self.settings.priorization == 'PER':
                    self.sampler.update_weights(self.sess)
                print('### EPOCH '+str(self.epoch_idx)+' ###')
                _current_state = self.nn.reset_state
                self.bar = Bar('Training', max = int((self.settings.num_batches_train)/self.settings.log_frequency), suffix = '[%(index)d/%(max)d] - %(elapsed)ds')
                while(not_done):
                    not_done, log, batchX, batchY, _current_state, _weights = self.sampler.next_train_batch(_current_state) 
                    if log:
                        _total_loss, _train_step, _current_state, _predictions_series, _summaries, _accuracy = self.train_log_op(batchX, batchY, _current_state, weights = _weights)
                        self.log_train(_summaries, _total_loss, _accuracy)
                    else:
                        _total_loss, _train_step, _current_state, _predictions_series = self.train_op(batchX, batchY, _current_state, weights = _weights)
                self.bar.finish()

            # TRAINING IS DONE
            self.save('final')
            print('TRAINING DONE ! ^_^')
        self.write_logs()

    def save(self, name):
        NN_save_name = os.path.join(self.settings.output_dir,name)
        self.saver.save(self.sess, NN_save_name)

    def write_logs(self):
        try:
            os.mkdir(self.settings.output_dir)
        except:
            pass

        self.train_acc_history = np.array(self.train_acc_history)
        self.train_loss_history = np.array(self.train_loss_history)
        self.train_time_history = np.array(self.train_time_history)
        self.train_step_history = np.array(self.train_step_history)
        self.val_acc_history = np.array(self.val_acc_history)
        self.val_loss_history = np.array(self.val_loss_history)
        self.val_time_history = np.array(self.val_time_history)
        self.val_step_history = np.array(self.val_step_history)
        self.val_ms_acc_history = np.array(self.val_ms_acc_history)
        self.val_ms_time_history = np.array(self.val_ms_time_history)
        self.val_ms_step_history = np.array(self.val_ms_step_history)

        self.train_acc_history = self.train_acc_history[...,np.newaxis]
        self.train_loss_history = self.train_loss_history[...,np.newaxis]
        self.train_time_history = self.train_time_history[...,np.newaxis]
        self.train_step_history = self.train_step_history[...,np.newaxis]
        self.val_acc_history = self.val_acc_history[...,np.newaxis]
        self.val_loss_history = self.val_loss_history[...,np.newaxis]
        self.val_time_history = self.val_time_history[...,np.newaxis]
        self.val_step_history = self.val_step_history[...,np.newaxis]
        self.val_ms_acc_history  = self.val_ms_acc_history[...,np.newaxis]
        self.val_ms_time_history = self.val_ms_time_history[...,np.newaxis]
        self.val_ms_step_history = self.val_ms_step_history[...,np.newaxis]

        train_TS = np.concatenate((self.train_time_history, self.train_step_history), axis = 1)
        train_acc_log = np.concatenate((train_TS, self.train_acc_history),axis = 1)
        train_loss_log = np.concatenate((train_TS, self.train_loss_history),axis = 1)

        val_TS = np.concatenate((self.val_time_history, self.val_step_history), axis = 1)
        val_acc_log = np.concatenate((val_TS, self.val_acc_history),axis = 1)
        val_loss_log = np.concatenate((val_TS, self.val_loss_history),axis = 1)
        
        val_ms_TS = np.concatenate((self.val_ms_time_history, self.val_ms_step_history), axis = 1)
        val_ms_acc_log = np.concatenate((val_ms_TS, self.val_ms_acc_history),axis = 1)

        np.save(self.settings.output_dir+'/train_acc.npy', train_acc_log)
        np.save(self.settings.output_dir+'/test_single_step_loss_log.npy', val_acc_log)
        np.save(self.settings.output_dir+'/test_multi_step_loss_log.npy', val_ms_acc_log)
        np.save(self.settings.output_dir+'/train_loss_log.npy', train_loss_log)
        np.save(self.settings.output_dir+'/val_loss.npy', val_loss_log)

    def restore(self):
        raise Exception('Not implemented')

TR = Train()
TR.train()
