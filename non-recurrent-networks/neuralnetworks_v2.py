import sys
import math
import os
import datetime
import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_squared_error as SK_MSE
#custom import
import samplers_v2
import readers_v2
import tf_graphv2
import settings_v2
import networklist_processor_v2

class UniformTraining:
    """
    Training container, extended for advanced priorization schemes.
    """
    def __init__(self):
        # Setting object
        self.sts = settings_v2.Settings()
        # Dataset object
        self.DS = None
        self.load_dataset()
        # Samplers
        self.SR = None
        self.load_sampler()
        # Model
        self.M = None
        self.load_model()
        # Train
        self.init_records()
        self.train()

    def load_dataset(self):
        #TODO use setting object in reader instead of feeding a shit tone of arguments
        self.DS = readers_v2.H5Reader(self.sts.train_dir, self.sts.test_dir,
                                      self.sts.val_dir, self.sts.input_dim,
                                      self.sts.output_dim, self.sts.cmd_dim,
                                      self.sts.sequence_length, self.sts.trajectory_length,
                                      self.sts.val_ratio, self.sts.test_ratio,
                                      self.sts.val_idx, self.sts.test_idx,
                                      ts_idx = self.sts.timestamp_idx)

    def load_sampler(self):
        #TODO integrate setting object in sampler
        self.SR = samplers_v2.UniformSampler(self.DS)

    def load_model(self):
        #TODO use setting object in the model to improve flexibility
        #TODO make networklist_processor less ugly
        self.M = networklist_processor_v2.get_graph(self.sts.model, self.sts.sequence_length,
                                                 self.sts.input_dim, self.sts.output_dim)
    
    def init_records(self):
        # Hard-Record settings (numpy array)
        self.train_logs = []
        self.test_logs = []
        self.test_logs_multi_step = []
        self.best_1s = np.inf
        self.best_ms = np.inf
        self.lw_ms = np.inf
        self.start_time = datetime.datetime.now()

    def saver_init_and_restore(self):
        # Saver init
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.train_writer = tf.summary.FileWriter(self.sts.tb_dir + '/train', self.sess.graph)
        self.test_writer = tf.summary.FileWriter(self.sts.tb_dir + '/test')
        if self.sts.restore:
            self.saver.restore(self.sess, self.sts.path_weight)

    def train_step(self, i):
        # Training on training set
        # Sample new training batch
        prct, batch_xs, batch_ys = self.SR.sample_train_batch(self.sts.batch_size)
        _ = self.sess.run(self.M.train_step, feed_dict = {self.M.x: batch_xs,
                                                          self.M.y: batch_ys,
                                                          self.M.weights: np.ones(self.sts.batch_size),
                                                          self.M.keep_prob: self.sts.dropout,
                                                          self.M.step: i,
                                                          self.M.is_training: True})

    def eval_on_train(self, i):
        # Single-Step performance evaluation on training set
        # Sample new large train batch
        #TODO REMOVE MAGIC NUMBER (1000) in eval train batch set default argument ?
        prct, batch_xs, batch_ys = self.SR.sample_eval_train_batch(1000)
        # Computes accuracy and loss + acquires summaries
        acc, loss, summaries = self.sess.run([self.M.acc_op, self.M.s_loss, self.M.merged],
                                        feed_dict = {self.M.x: batch_xs,
                                                     self.M.y: batch_ys,
                                                     self.M.weights: np.ones(batch_xs.shape[0]),
                                                     self.M.keep_prob: self.sts.dropout,
                                                     self.M.step: i,
                                                     self.M.is_training: False})
        # Update hard-logs
        elapsed_time = (datetime.datetime.now() - self.start_time).total_seconds()
        self.train_logs.append([i] + [elapsed_time] + list(acc))
        # Write tensorboard logs
        self.train_writer.add_summary(summaries, i)

    def eval_on_validation_single_step(self, i):
        # Single-Step performance evaluation on validation set
        # Sample a batch as the whole of the validation set
        # TODO set option full dataset or given batch size
        prct, batch_xs , batch_ys = self.SR.sample_val_batch() 
        # Computes accuracy and loss + acquires summaries
        acc, loss, summaries = self.sess.run([self.M.acc_op, self.M.s_loss, self.M.merged],
                                        feed_dict = {self.M.x: batch_xs,
                                                     self.M.y: batch_ys,
                                                     self.M.weights: np.ones(batch_xs.shape[0]),
                                                     self.M.keep_prob: self.sts.dropout,
                                                     self.M.step: i,
                                                     self.M.is_training: False})
        # Update Single-Step hard-logs
        elapsed_time = (datetime.datetime.now() - self.start_time).total_seconds()
        self.test_logs.append([i] + [elapsed_time]+list(acc))
        # Update inner variables and saves best model weights
        avg = np.mean(acc)
        if  avg < self.best_1s:
            self.best_1s = avg
            NN_save_name = os.path.join(self.sts.model_ckpt,'Best_1S')
            self.saver.save(self.sess, NN_save_name)
        # Write tensorboard logs
        self.test_writer.add_summary(summaries, i)
        # Return accuracy for console display
        return acc
        
    def eval_on_validation_multi_step(self, i):
        # Multi step performance evaluation on validation set
        predictions = []
        # Sample trajectory batch out of the evaluation set
        batch_x, batch_y = self.SR.sample_val_trajectory()
        # Take only the first elements of the trajectory
        full = batch_x[:,:self.sts.sequence_length,:]
        # Run iterative predictions
        for k in range(self.sts.sequence_length, self.sts.sequence_length+self.sts.trajectory_length - 1):
            # Get predictions
            pred = self.sess.run(self.M.y_, feed_dict = {self.M.x: full,
                                                    self.M.keep_prob:self.sts.dropout,
                                                    self.M.weights: np.ones(full.shape[0]),
                                                    self.M.is_training: False,
                                                    self.M.step: i})
            predictions.append(np.expand_dims(pred, axis=1))
            # Remove first elements of old batch add predictions
            # concatenated with the next command input
            cmd = batch_x[:, k+1, -self.sts.cmd_dim:]
            new = np.concatenate((pred, cmd), axis=1)
            new = np.expand_dims(new, axis=1)
            old = full[:,1:,:]
            full = np.concatenate((new,old), axis=1)
        predictions = np.concatenate(predictions, axis = 1)
        # Compute per variable error
        error_x = [SK_MSE(predictions[:,:,i], batch_y[:,:-1,i]) for i in range(predictions.shape[-1])]
        worse = np.max(error_x)
        avg = np.mean(error_x)
        # Update multistep hard-logs
        elapsed_time = (datetime.datetime.now() - self.start_time).total_seconds()
        self.test_logs_multi_step.append([i] + [elapsed_time] + list(error_x) + [worse])
        # Update inner variable
        if avg < self.best_ms:
            self.best_ms = avg
            NN_save_name = os.path.join(self.sts.output_dir,'Best_MS')
            self.saver.save(self.sess, NN_save_name)
        if worse < self.lw_ms:
            self.lw_ms = worse
            NN_save_name = os.path.join(self.sts.output_dir,'Least_Worse_MS')
            self.saver.save(self.sess, NN_save_name)
        return error_x, worse

    def display(self, i, acc, worse, ms_acc):
        # Simple console display every N iterations
        print('Step: ', str(i), ', 1s acc:', str(acc), ', ', str(self.sts.trajectory_length),
                       's worse acc: ', str(worse), ', ',
                       str(self.sts.trajectory_length), 's avg acc: ', str(ms_acc))

    def dump_logs(self):
        # Save model weights at the end of training
        NN_save_name = os.path.join(self.sts.output_dir,'final_NN')
        self.saver.save(self.sess, NN_save_name)
        # Display training statistics
        print('#### TRAINING DONE ! ####')
        print('Best single-step-Accuracy reach for: ' + str(self.best_1s))
        print('Best multi-steps-Accuracy reach for: ' + str(self.best_ms))
        print('Least Worse Accuracy reach for: ' + str(self.lw_ms))
        # Write hard-logs as numpy arrays
        np.save(self.sts.output_dir + "/train_loss_log.npy", np.array(self.train_logs))
        np.save(self.sts.output_dir + "/test_single_step_loss_log.npy", np.array(self.test_logs))
        np.save(self.sts.output_dir + "/test_multi_step_loss_log.npy", np.array(self.test_logs_multi_step))
        np.save(self.sts.output_dir + "/means.npy", self.DS.mean)
        np.save(self.sts.output_dir + "/std.npy", self.DS.std)

    def train(self):
        with tf.Session() as self.sess:
            self.saver_init_and_restore()
            for i in range(self.sts.max_iterations):
                self.train_step(i)
                if i%10 == 0:
                    self.eval_on_train(i)
                if i%50 == 0:
                    acc = self.eval_on_validation_single_step(i)
                    acc_t, worse = self.eval_on_validation_multi_step(i)
                if i%250 == 0:
                    self.display(i, acc, worse, acc_t)
            self.dump_logs()
"""
class PERTraining(UniformTraining):
    def __init__(self):
        super(PERTraining, self).__init__()
        # Load Sampler
        self.sampler = None
        self.load_sampler()

    def load_sampler(self):
        #TODO integrate setting object in sampler
        self.SR = sampler_v2.PERSampler(self.DS, alpha = self.sts.alpha, beta = self.sts.beta, e = self.sts.epsilon)

    def train_per():
        w_i, batch_xs, batch_ys = self.SR.sample_train_batch(self.batch_size)
        _ = sess.run(self.M.train_step, feed_dict = {self.M.x:batch_xs,
                                                     self.M.y:batch_ys,
                                                     self.M.weights: w_i,
                                                     self.M.keep_prob: self.sts.dropout,
                                                     self.M.step: i,
                                                     self.M.is_training: True})

class GradTraining(UniformTraining):
    def __init__(self):
        super(GradTraining, self).__init__()

    def load_sampler(self):
        #TODO integrate setting object in sampler
        self.SR = sampler_v2.GradSampler(self.DS)

    def train_grad():
        w_i_scoring = np.ones(self.ss_bs)
        prct, ss_batch_xs, ss_batch_ys = sampler.sample_superbatch(self.ss_bs)
        score = sess.run(grads, feed_dict = {x:ss_batch_xs, y:ss_batch_ys, keep_prob: 1.0, step: i, weights:w_i_scoring})
        idxs, w_i = sampler.sample_scored_batch(self.ss_bs, self.bs, score)
        batch_xs, batch_ys = sampler.train_batch_RNN(idxs)
        _ = sess.run(train_step, feed_dict = {x:batch_xs, y:batch_ys, keep_prob: 1.0, step: i, weights:w_i})
"""
if __name__ == '__main__':
    UT = UniformTraining()
