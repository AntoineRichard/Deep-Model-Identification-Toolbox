import sys
import math
import os
import datetime
import tensorflow as tf
import numpy as np
import h5py as h5
from sklearn.metrics import mean_squared_error as SK_MSE
#custom import
import samplers#class to handle the data sampling
import hdf5_reader
import nntools
import tf_graph
import networklist_processor

class NN:

    """
    class to define the architecture and the training scheme of a neural networks
    """
    def __init__(self,args):
        self.list_dataset_path = args.dataset
        self.output_path = args.output
        self.split_test = args.split_test #number/index (?) of split of the total dataset for the k fold testing
        if args.split_val == 42:
            self.split_val = None
        else:
            self.split_val = args.split_val #number/index (?) of split of the training subdataset for cross-validation
        self.bs = args.batch_size
        self.sbs = args.super_batch_size
        self.iterations = args.iterations # number of iteration for NN training
        self.input_history = args.used_points #number of step used for prediciton <=> arma order
        self.output_forecast = args.pred_points # number of step the model predict
        self.undersampling = args.frequency # undersampling factor
        self.test_window = args.test_window # window soze for multistep prediction
        self.multistep_test_number = args.test_iter # number of multistep prediction test done
        self.alpha = args.alpha # PER alpha hyperparameter
        self.beta = args.beta # PER beta hyperparameter
        self.per_iter = args.experience_replay # PER iteration number
        self.network_type = args.model_type #type of the model one of : classic, loss
        self.optimisation_type = args.optimisation_type #type of the model one of : none, grad, PER
        self.input_dim = args.input_state_dim #dimention of input state
        self.output_dim = args.output_state_dim #dimention of output state
        self.cmd_dim = args.cmd_dim #dimention of output state

        self.graph = networklist_processor.get_graph(self.network_type, self.input_history, self.input_dim,self.output_forecast, self.output_dim)
        self.ds_type = args.ds_type
        try:
            os.mkdir(self.output_path)
        except:
            pass
    
        def train_grad():
            w_i_scoring = np.ones(self.ss_bs)
            prct, ss_batch_xs, ss_batch_ys = sampler.sample_superbatch(self.sbs)
            score = sess.run(grads, feed_dict = {x:ss_batch_xs, y:ss_batch_ys, keep_prob: 1.0, step: i, weights:w_i_scoring})
            idxs, w_i = sampler.sample_scored_batch(self.ss_bs, self.bs, score)
            batch_xs, batch_ys = sampler.train_batch_RNN(idxs)
            _ = sess.run(train_step, feed_dict = {x:batch_xs, y:batch_ys, keep_prob: 1.0, step: i, weights:w_i})

        def train_per():
            w_i = np.ones(self.bs)
            _ = sess.run(self.graph_mlp.train_step, feed_dict = {self.graph_mlp.x:batch_xs, self.graph_mlp.y:batch_ys, self.graph_mlp.keep_prob: 0.75, self.graph_mlp.step: i, self.graph_mlp.weights: w_i})


        def train_uniform():

    def train_classic(self):
        dataset = hdf5_reader.H5Reader(self.list_dataset_path[0], self.list_dataset_path[1], self.list_dataset_path[2], self.input_dim, self.output_dim, self.cmd_dim, self.input_history, self.test_window, ts_idx=0)

        np.save(self.output_path + "/means.npy", dataset.mean)
        np.save(self.output_path + "/std.npy", dataset.std)

        sampler = samplers.UniformSampler(dataset)
        tfconfig = tf.ConfigProto(device_count = {'GPU': 0})

        # Saves training parameters
        train_logs = []
        test_logs = []
        test_logs_multi_step = []
        best_1s_nn = 100000.0
        best_ms_nn = 100000.0
        least_worse_ms_nn = 100000.0
        start_time = datetime.datetime.now()

        train_eval = sampler.sample(1000)
        with self.graph_mlp.g.as_default():
            with tf.Session(config = tfconfig) as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
                saver = tf.train.Saver()
                # ADD RESTORE OPTION

                w_i = np.ones(self.bs)
                for i in range(self.iterations):
                    prct, batch_xs, batch_ys = sampler.sample_train_batch(bs)
                    _ = sess.run(self.graph_mlp.train_step, feed_dict = {self.graph_mlp.x:batch_xs, self.graph_mlp.y:batch_ys,  self.graph_mlp.weights:w_i, self.graph_mlp.keep_prob: 0.75, self.graph_mlp.step: i})

                    if i%10 == 0:
                        prct, batch_xs, batch_ys = sampler.sample_eval_train_batch(1000)
                        acc, pred = sess.run([self.graph_mlp.acc_op, self.graph_mlp.y_], feed_dict = {self.graph_mlp.x:batch_xs, self.graph_mlp.y:batch_ys, self.graph_mlp.keep_prob: 0.75, self.graph_mlp.step: i})
                        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
                        train_logs.append([i] + [elapsed_time] + list(acc))

                        avg = np.mean(acc)
                        if  avg < best_1s_nn:
                            best_1s_nn = avg
                            NN_save_name = self.output_path + '/best_1S_NN'
                            saver.save(sess, NN_save_name)

                    if i%50 == 0:
                        # Single step performance evaluation on test set
                        batch_x , batch_y = sampler.sample_val_batch()
                        acc = sess.run(self.graph_mlp.acc_op, feed_dict = {self.graph_mlp.x:batch_x, self.graph_mlp.y:batch_y, self.graph_mlp.keep_prob:1.0, self.graph_mlp.step:i})
                        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
                        test_logs.append([i] + [elapsed_time]+list(acc))
                        
                        # Multi step performance evaluation
                        worse = 0
                        avg = []
                        trajectories = []
                        GT = []
                        #for j in range(self.multistep_test_number):
                        batch_x, batch_y = sampler.sample_val_trajectory()
                        full = batch_x[:,:self.input_history,:]
                        predictions = []

                        for k in range(self.input_history, self.input_history+self.test_window - 1):
                            pred = sess.run(self.graph_mlp.y_,feed_dict = {self.graph_mlp.x: full, self.graph_mlp.keep_prob:1.0, self.graph_mlp.step: i})
                            predictions.append(np.expand_dims(pred, axis=1))
                            cmd = batch_x[:, k+1, -self.cmd_dim:]
                            new = np.concatenate((pred, cmd), axis=1)
                            new = np.expand_dims(new, axis=1)
                            old = full[:,1:,:]
                            full = np.concatenate((new,old), axis=1)
                        predictions = np.concatenate(predictions,axis = 1)
                        error_x1 = SK_MSE(predictions[:,:,0],batch_y[:,:-1,0])
                        error_x2 = SK_MSE(predictions[:,:,1],batch_y[:,:-1,1])
                        error_x3 = SK_MSE(predictions[:,:,2],batch_y[:,:-1,2])
                        trajectories.append(predictions)
                        GT.append(batch_y[:-1,:3])
                        err = [error_x1, error_x2, error_x3]
                        if np.mean(err) > worse:
                            worse = np.mean(err)
                        avg.append(err)

                        err = np.mean(avg,axis=0)
                        avg = np.mean(avg)
                        if avg < best_ms_nn:
                            trajectories = np.array(trajectories)
                            GT = np.array(GT)
                            np.save(self.output_path + '/best_trajectories_predicted',trajectories)
                            np.save(self.output_path + '/best_trajectories_groundtruth',GT)
                            best_ms_nn = avg
                            NN_save_name = self.output_path + '/best_NN'
                            saver.save(sess, NN_save_name)
                        if worse < least_worse_ms_nn:
                            least_worse_ms_nn = worse
                            NN_save_name = self.output_path + '/least_worse_NN'
                            saver.save(sess, NN_save_name)
                        # Multi step performance evaluation on test set
                        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
                        test_logs_multi_step.append([i] + [elapsed_time] + list(err) + [worse])
                        if i%250 == 0:
                            print('Step: ' + str(i) + ', 1s acc:' + str(acc) + ', ' + str(self.test_window) + 's worse acc: ' + str(worse) + ', ' + str(self.test_window) + 's avg acc: ' + str(err))


                NN_save_name = self.output_path + '/final_NN'
                saver.save(sess, NN_save_name)

                print('#### TRAINING DONE ! ####')
                print('Best single-step-Accuracy reach for: ' + str(best_1s_nn))
                print('Best multi-steps-Accuracy reach for: ' + str(best_ms_nn))
                print('Least Worse Accuracy reach for: ' + str(least_worse_ms_nn))
                np.save(self.output_path + "/train_loss_log.npy", np.array(train_logs))
                np.save(self.output_path + "/test_single_step_loss_log.npy", np.array(test_logs))
                np.save(self.output_path + "/test_multi_step_loss_log.npy", np.array(test_logs_multi_step))

    def train(self):
        #calling the rigth train using self.model_type.
        #some nice patern use would be nice here ...
        # ... todo
        #if self.network_type == 'classic' and self.optimisation_type == 'none' :
        if self.ds_type == 'boat':
            self.train_classic()
        elif self.ds_type == 'asctec': 
           self.train_classic_ethz()
        elif self.ds_type == 'drone': 
           self.train_classic_drone()

        #else:
            #error = "type not supported : " + self.network_type + ' ' + self.optimisation_type
            #raise Exception(error)#better to use RunTimeError ?



