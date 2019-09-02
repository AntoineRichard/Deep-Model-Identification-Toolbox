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
        self.graph_mlp = networklist_processor.get_graph(self.network_type, self.input_history, self.input_dim,self.output_forecast, self.output_dim)
        self.ds_type = args.ds_type
        try:
            os.mkdir(self.output_path)
        except:
            pass
        #if self.network_type == 'classic':
        #    self.graph_mlp = tf_graph.GraphMLP(self.input_history,self.input_dim,self.output_forecast,self.output_dim)
        #elif self.network_type == 'loss':
        #    print("loss graph")
        #    self.graph_mlp_loss = tf_graph.GraphLoss(self.input_history,self.input_dim,self.output_forecast,self.output_dim, self.cmd_dim)
        #else:
        #    raise Exception("network type not supported : " + self.network_type)

# @TODO : 1s/5s/ns as parameter
# @TODO : grad/per/none as parameter
    def train_grad_1s(self):

        dataset = hdf5_reader.ReaderDrone(self.list_dataset_path, self.input_history, self.input_dim,self.output_forecast, self.output_dim, self.cmd_dim, self.split_test, self.split_val)#do all the test/valid splitting

        np.save(self.output_path + "/means.npy", dataset.mean)
        np.save(self.output_path + "/std.npy", dataset.std)



        grad_sampler = samplers.LossSampler(dataset,self.multistep_test_number)#loss sampler ? legacy name ?

        # Session configuration : train only on CPU
        tfconfig = tf.ConfigProto(device_count = {'GPU': 0})

        # Saves training parameters
        train_logs = []
        test_logs = []
        test_logs_multi_step = []
        best_nn = 100000.0
        least_worse_nn = 10000.0
        start_time = datetime.datetime.now()

        with self.graph_mlp.g.as_default():
            with tf.Session(config = tfconfig) as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
                saver = tf.train.Saver()

                w_i = np.ones(self.bs)

                for i in range(self.iterations):
                    w_i_scoring = np.ones(self.sbs)
                    samples = grad_sampler.sample_superbatch(self.sbs)
                    ss_i, ss_l = grad_sampler.train_batch(samples)
                    score = sess.run(self.graph_mlp.grad, feed_dict = {self.graph_mlp.x:ss_i, self.graph_mlp.y:ss_l, self.graph_mlp.keep_prob: 0.75, self.graph_mlp.step: i, self.graph_mlp.weights:w_i_scoring})
                    idxs, probs = grad_sampler.sample_scored_batch(self.sbs,self.bs,score,samples)
                    w_i = 1/probs
                    batch_xs, batch_ys = grad_sampler.train_batch(idxs)
                    _ = sess.run(self.graph_mlp.train_step, feed_dict = {self.graph_mlp.x:batch_xs, self.graph_mlp.y:batch_ys, self.graph_mlp.keep_prob: 0.75, self.graph_mlp.step: i, self.graph_mlp.weights:w_i})

                    if i%10 == 0:
                        # Single step performance evaluation on train set
                        batch_x , batch_y = dataset.getTrain(range(1000))
                        acc = sess.run(self.graph_mlp.acc_op, feed_dict = {self.graph_mlp.x:batch_xs, self.graph_mlp.y:batch_ys, self.graph_mlp.keep_prob: 0.75, self.graph_mlp.step: i})
                        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
                        train_logs.append([i,elapsed_time,acc])

                    if i%50 == 0:
                        # Single step performance evaluation on test set
                        batch_x , batch_y = dataset.getVal(range(dataset.val_size))
                        acc = sess.run(self.graph_mlp.acc_op,feed_dict = {self.graph_mlp.x:batch_x, self.graph_mlp.y:batch_y, self.graph_mlp.keep_prob:1.0, self.graph_mlp.step:i})
                        test_logs.append([i,acc])
                        worse = 0
                        avg = []
                        for j in range(self.multistep_test_number):
                            #@TODO
                            batch_x, batch_y = grad_sampler.get_random_val_batch(self.test_window)

                            full = batch_x[0,:self.input_history * self.input_dim][np.newaxis,...]#add a axis to the array
                            predictions = []

                            for k in range(self.test_window - 1):
                                pred = sess.run(self.graph_mlp.y_,feed_dict = {self.graph_mlp.x: full, self.graph_mlp.keep_prob:1.0, self.graph_mlp.step: i})
                                predictions.append([pred[0,0],pred[0,1],pred[0,2]])
                                cmd = batch_x[k+1,-self.cmd_dim:][np.newaxis,...]
                                new = np.concatenate((pred,cmd),axis=1)
                                old = full[0,:-self.input_dim][np.newaxis,...]
                                full = np.concatenate((new,old),axis=1)
                            predictions = np.array(predictions)
                            error_x1 = SK_MSE(predictions[:,0],batch_y[:-1,0])
                            error_x2 = SK_MSE(predictions[:,1],batch_y[:-1,1])
                            error_x3 = SK_MSE(predictions[:,2],batch_y[:-1,2])
                            if np.mean([error_x1,error_x2,error_x3]) > worse:
                                worse = np.mean([error_x1, error_x2, error_x3])
                            avg.append(np.mean([error_x1, error_x2, error_x3]))

                        avg = np.mean(avg)
                        if avg < best_nn:
                            best_nn = avg
                            NN_save_name = self.output_path + '/best_NN'
                            saver.save(sess, NN_save_name)
                        if worse < least_worse_nn:
                            least_worse_nn = worse
                            NN_save_name = self.output_path + '/least_worse_NN'
                            saver.save(sess, NN_save_name)
                        # Multi step performance evaluation on test set
                        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
                        test_logs_multi_step.append([i,elapsed_time,avg,worse])
                        if i%250 == 0:
                            print('Step: ' + str(i) + ', 1s acc:' + str(acc) + ', ' + str(self.test_window) + 's worse acc: ' + str(worse) + ', ' + str(self.test_window) + 's avg acc: ' + str(avg))
                        batch_x , batch_y = grad_sampler.next_test_batch(self.bs)

                NN_save_name = self.output_path + '/final_NN'
                saver.save(sess, NN_save_name)

                print('#### TRAINING DONE ! ####')
                print('Best Accuracy reach for: ' + str(best_nn))
                print('Least Worse Accuracy reach for: ' + str(least_worse_nn))
                np.save(self.output_path + "/train_loss_log.npy", np.array(train_logs))
                np.save(self.output_path + "/test_single_step_loss_log.npy", np.array(test_logs))
                np.save(self.output_path + "/test_multi_step_loss_log.npy", np.array(test_logs_multi_step))

                np.save(self.output_path + "/result.npy", predictions)



    def train_grad_5s(self):

        dataset = hdf5_reader.ReaderDrone(self.list_dataset_path, self.input_history, self.input_dim,self.output_forecast, self.output_dim, self.cmd_dim, self.split_test, self.split_val)#do all the test/valid splitting

        np.save(self.output_path + "/means_x.npy", dataset.mean_train_x)
        np.save(self.output_path + "/std_x.npy", dataset.std_train_x)

        #note : in regression normalisation data for y should be redundant with x
        np.save(self.output_path + "/means_y.npy", dataset.mean_train_y)
        np.save(self.output_path + "/std_y.npy", dataset.std_train_y)


        sampler = samplers.LossSampler(dataset,self.multistep_test_number)

        # Session configuration : train only on CPU
        tfconfig = tf.ConfigProto(device_count = {'GPU': 0})

        # Saves training parameters
        train_logs = []
        test_logs = []
        test_logs_multi_step = []
        best_nn = 100000.0
        least_worse_nn = 10000.0
        start_time = datetime.datetime.now()

        #tensor def for shorter writung and better readibility :
        train_step = self.graph_mlp_loss.train_step
        x = self.graph_mlp_loss.x
        y = self.graph_mlp_loss.y
        keep_prob = self.graph_mlp_loss.keep_prob
        y_ = self.graph_mlp_loss.y_
        acc_op = self.graph_mlp_loss.acc_op
        grads = self.graph_mlp_loss.grads
        step = self.graph_mlp_loss.step
        weights = self.graph_mlp_loss.weights

        with self.graph_mlp_loss.g.as_default():

            with tf.Session(config = tfconfig) as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
                saver = tf.train.Saver()

                for i in range(self.iterations):
                    w_i_scoring = np.ones(self.sbs)
                    samples = sampler.sample_superbatch(self.sbs)
                    ss_i, ss_l = sampler.train_batch_RNN(samples)
                    score = sess.run(grads, feed_dict = {x:ss_i, y:ss_l, keep_prob: 0.75, step: i, weights:w_i_scoring})
                    idxs, probs = sampler.sample_scored_batch(self.sbs,self.bs,score,samples)
                    w_i = 1/probs
                    batch_xs, batch_ys = sampler.train_batch_RNN(idxs)
                    _ = sess.run(train_step, feed_dict = {x:batch_xs, y:batch_ys, keep_prob: 0.75, step: i, weights:w_i})

                    if i%10 == 0:
                        # Single step performance evaluation on train set
                        batch_x , batch_y = dataset.getTrainRNN(range(1000))
                        acc = sess.run(acc_op, feed_dict = {x:batch_xs, y:batch_ys, keep_prob: 0.75, step: i, weights:w_i})
                        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
                        train_logs.append([i,elapsed_time,acc])

                    if i%50 == 0:
                        # Single step performance evaluation on test set
                        batch_x , batch_y = dataset.getValRNN(range(dataset.val_size))
                        acc = sess.run(acc_op,feed_dict = {x:batch_x, y:batch_y, keep_prob:1.0, step:i, weights:w_i})
                        test_logs.append([i,acc])
                        worse = 0
                        avg = []
                        for j in range(self.multistep_test_number):
                            batch_x, batch_y = sampler.get_random_val_batch_RNN(self.test_window)

                            full = batch_x[0,:self.input_history * self.input_dim][np.newaxis,...]
                            predictions = []

                            for k in range(self.test_window - 1):
                                pred = sess.run(y_, feed_dict = {x: full, keep_prob:1.0, step: i})
                                predictions.append([pred[0,0], pred[0,1], pred[0,2], pred[0,3]])
                                cmd = batch_x[k + 1, -self.cmd_dim:][np.newaxis,...]
                                new = np.concatenate((pred, cmd), axis=1)
                                old = full[0,self.input_dim:][np.newaxis,...]
                                full = np.concatenate((old, new), axis=1)
                            predictions = np.array(predictions)
                            error_x1 = SK_MSE(predictions[:,0],batch_y[:-1,0])
                            error_x2 = SK_MSE(predictions[:,1],batch_y[:-1,1])
                            error_x3 = SK_MSE(predictions[:,2],batch_y[:-1,2])
                            error_x4 = SK_MSE(predictions[:,3],batch_y[:-1,3])
                            if np.mean([error_x1,error_x2,error_x3,error_x4]) > worse:
                                worse = np.mean([error_x1, error_x2, error_x3, error_x4])
                            avg.append(np.mean([error_x1, error_x2, error_x3, error_x4]))

                        avg = np.mean(avg)
                        if avg < best_nn:
                            best_nn = avg
                            NN_save_name = self.output_path + '/best_NN'
                            saver.save(sess, NN_save_name)
                        if worse < least_worse_nn:
                            least_worse_nn = worse
                            NN_save_name = self.output_path + '/least_worse_NN'
                            saver.save(sess, NN_save_name)
                        # Multi step performance evaluation on test set
                        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
                        test_logs_multi_step.append([i, elapsed_time, avg, worse])
                        if i%250 == 0:
                            print('Step: ' + str(i) + ', 1s acc:' + str(acc) + ', ' + str(self.test_window) + 's worse acc: ' + str(worse) + ', ' + str(self.test_window) + 's avg acc: ' + str(avg))

                NN_save_name = self.output_path + '/final_NN'
                saver.save(sess, NN_save_name)

                print('#### TRAINING DONE ! ####')
                print('Best Accuracy reach for: ' + str(best_nn))
                print('Least Worse Accuracy reach for: ' + str(least_worse_nn))
                np.save(self.output_path + "/train_loss_log.npy", np.array(train_logs))
                np.save(self.output_path + "/test_single_step_loss_log.npy", np.array(test_logs))
                np.save(self.output_path + "/test_multi_step_loss_log.npy", np.array(test_logs_multi_step))

                np.save(self.output_path + "/result.npy", predictions)


    def train_per(self):
        dataset = hdf5_reader.ReaderDrone(self.list_dataset_path, self.input_history, self.input_dim,self.output_forecast, self.output_dim, self.cmd_dim, self.split_test, self.split_val)#do all the test/valid splitting

        np.save(self.output_path + "/means_x.npy", dataset.mean_train_x)
        np.save(self.output_path + "/std_x.npy", dataset.std_train_x)

        #note : in regression normalisation data for y should be redundant with x
        np.save(self.output_path + "/means_y.npy", dataset.mean_train_y)
        np.save(self.output_path + "/std_y.npy", dataset.std_train_y)


        sampler = samplers.PrioritizingSampler(dataset,self.multistep_test_number,alpha = self.alpha, beta = self.beta)

        # Session configuration : train only on CPU
        tfconfig = tf.ConfigProto(device_count = {'GPU': 0})

        # Saves training parameters
        train_logs = []
        test_logs = []
        test_logs_multi_step = []
        best_nn = 100000.0
        least_worse_nn = 10000.0
        start_time = datetime.datetime.now()

        with self.graph_mlp.g.as_default():
            with tf.Session(config = tfconfig) as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
                saver = tf.train.Saver()
                for i in range(self.iterations):
                    w_i = np.ones(self.bs)
                    batch_xs, batch_ys = sampler.next_train_batch(self.bs)
                    _ = sess.run(self.graph_mlp.train_step, feed_dict = {self.graph_mlp.x:batch_xs, self.graph_mlp.y:batch_ys, self.graph_mlp.keep_prob: 0.75, self.graph_mlp.step: i, self.graph_mlp.weights: w_i})
                    if i%10 == 0:
                        # Single step performance evaluation on train set
                        batch_x , batch_y = dataset.getTrain(range(1000))
                        acc = sess.run(self.graph_mlp.acc_op, feed_dict = {self.graph_mlp.x:batch_xs, self.graph_mlp.y:batch_ys, self.graph_mlp.keep_prob: 0.75, self.graph_mlp.step: i})
                        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
                        train_logs.append([i,elapsed_time,acc])

                    if i%50 == 0:
                        # Single step performance evaluation on test set
                        batch_x , batch_y = dataset.getVal(range(dataset.val_size))
                        acc = sess.run(self.graph_mlp.acc_op, feed_dict = {self.graph_mlp.x:batch_x, self.graph_mlp.y:batch_y, self.graph_mlp.keep_prob:1.0, self.graph_mlp.step:i})
                        test_logs.append([i,acc])
                        worse = 0
                        avg = []
                        for j in range(self.multistep_test_number):
                            batch_x, batch_y = sampler.get_random_val_batch(self.test_window)

                            full = batch_x[0,:self.input_history * self.input_dim][np.newaxis,...]
                            predictions = []
                            for k in range(self.test_window - 1):
                                pred = sess.run(self.graph_mlp.y_,feed_dict = {self.graph_mlp.x: full, self.graph_mlp.keep_prob:1.0, self.graph_mlp.step: i})
                                predictions.append([pred[0,0],pred[0,1],pred[0,2],pred[0,3]])
                                cmd = batch_x[k+1, -self.cmd_dim:][np.newaxis,...]
                                new = np.concatenate((pred, cmd), axis=1)
                                old = full[0,self.input_dim:][np.newaxis,...]
                                full = np.concatenate((old,new), axis=1)
                            predictions = np.array(predictions)
                            error_x1 = SK_MSE(predictions[:,0],batch_y[:-1,0])
                            error_x2 = SK_MSE(predictions[:,1],batch_y[:-1,1])
                            error_x3 = SK_MSE(predictions[:,2],batch_y[:-1,2])
                            error_x4 = SK_MSE(predictions[:,3],batch_y[:-1,3])
                            if np.mean([error_x1,error_x2,error_x3,error_x4]) > worse:
                                worse = np.mean([error_x1, error_x2, error_x3, error_x4])
                            avg.append(np.mean([error_x1, error_x2, error_x3, error_x4]))

                        avg = np.mean(avg)
                        if avg < best_nn:
                            best_nn = avg
                            NN_save_name = self.output_path + '/best_NN'
                            saver.save(sess, NN_save_name)
                        if worse < least_worse_nn:
                            least_worse_nn = worse
                            NN_save_name = self.output_path + '/least_worse_NN'
                            saver.save(sess, NN_save_name)
                        # Multi step performance evaluation on test set
                        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
                        test_logs_multi_step.append([i, elapsed_time, avg, worse])
                        if i%250 == 0:
                            print('Step: ' + str(i) + ', 1s acc:' + str(acc) + ', ' + str(self.test_window) + 's worse acc: ' + str(worse) + ', '+str(self.test_window) + 's avg acc: ' + str(avg))

                # Update
                for k in range(self.per_iter):

                    ex, ey = sampler.DS.getTrain(range(0,sampler.DS.train_size))
                    predictions = sess.run(self.graph_mlp.s_loss,feed_dict = {self.graph_mlp.x:ex, self.graph_mlp.y:ey, self.graph_mlp.keep_prob:1.0})
                    sampler.prioritize(predictions)

                    for i in range((k + 1) * self.iterations, (k + 2) * self.iterations):
                        batch_xs, batch_ys, w_i = sampler.next_train_batch_prioritized(self.bs)
                        score = sess.run(self.graph_mlp.train_step, feed_dict = {self.graph_mlp.x:batch_xs, self.graph_mlp.y:batch_ys, self.graph_mlp.keep_prob: 0.75, self.graph_mlp.step: i, self.graph_mlp.weights: w_i})
                        if i%10 == 0:
                            # Single step performance evaluation on train set
                            batch_x , batch_y = dataset.getTrain(range(1000))
                            acc = sess.run(self.graph_mlp.acc_op, feed_dict = {self.graph_mlp.x:batch_xs, self.graph_mlp.y:batch_ys, self.graph_mlp.keep_prob: 0.75, self.graph_mlp.step: i})
                            elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
                            train_logs.append([i, elapsed_time, acc])

                        if i%50 == 0:
                            # Single step performance evaluation on test set
                            batch_x , batch_y = dataset.getVal(range(dataset.val_size))
                            acc = sess.run(self.graph_mlp.acc_op, feed_dict = {self.graph_mlp.x:batch_x, self.graph_mlp.y:batch_y, self.graph_mlp.keep_prob:1.0, self.graph_mlp.step:i})
                            test_logs.append([i,acc])
                            worse = 0
                            avg = []
                            for j in range(self.multistep_test_number):
                                batch_x, batch_y = sampler.get_random_val_batch(self.test_window)

                                full = batch_x[0,:self.input_history * self.input_dim][np.newaxis,...]
                                predictions = []
                                for k in range(self.test_window - 1):
                                    pred = sess.run(self.graph_mlp.y_,feed_dict = {self.graph_mlp.x: full, self.graph_mlp.keep_prob:1.0, self.graph_mlp.step: i})
                                    predictions.append([pred[0,0],pred[0,1],pred[0,2],pred[0,3]])
                                    cmd = batch_x[k+1, -self.cmd_dim:][np.newaxis,...]
                                    new = np.concatenate((pred, cmd), axis=1)
                                    old = full[0,self.input_dim:][np.newaxis,...]
                                    full = np.concatenate((old,new), axis=1)
                                predictions = np.array(predictions)
                                error_x1 = SK_MSE(predictions[:,0],batch_y[:-1,0])
                                error_x2 = SK_MSE(predictions[:,1],batch_y[:-1,1])
                                error_x3 = SK_MSE(predictions[:,2],batch_y[:-1,2])
                                error_x4 = SK_MSE(predictions[:,3],batch_y[:-1,3])
                                if np.mean([error_x1,error_x2,error_x3,error_x4]) > worse:
                                    worse = np.mean([error_x1, error_x2, error_x3, error_x4])
                                avg.append(np.mean([error_x1, error_x2, error_x3, error_x4]))

                            avg = np.mean(avg)
                            if avg < best_nn:
                                best_nn = avg
                                NN_save_name = self.output_path + '/best_NN'
                                saver.save(sess, NN_save_name)
                            if worse < least_worse_nn:
                                least_worse_nn = worse
                                NN_save_name = self.output_path + '/least_worse_NN'
                                saver.save(sess, NN_save_name)
                            # Multi step performance evaluation on test set
                            elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
                            test_logs_multi_step.append([i, elapsed_time, avg, worse])
                            if i%250 == 0:
                                print('Step: ' + str(i) + ', 1s acc:' + str(acc) + ', ' + str(self.test_window)+'s worse acc: ' + str(worse) + ', ' + str(self.test_window) + 's avg acc: ' + str(avg))

                NN_save_name = self.output_path + '/final_NN'
                saver.save(sess, NN_save_name)

                print('#### TRAINING DONE ! ####')
                print('Best Accuracy reach for: ' + str(best_nn))
                print('Least Worse Accuracy reach for: ' + str(least_worse_nn))
                np.save(self.output_path + "/train_loss_log.npy", np.array(train_logs))
                np.save(self.output_path + "/test_single_step_loss_log.npy", np.array(test_logs))
                np.save(self.output_path + "/test_multi_step_loss_log.npy", np.array(test_logs_multi_step))

    def train_per_5s(self):
        dataset = hdf5_reader.ReaderDrone(self.list_dataset_path, self.input_history, self.input_dim,self.output_forecast, self.output_dim, self.cmd_dim, self.split_test, self.split_val)#do all the test/valid splitting

        np.save(self.output_path + "/means_x.npy", dataset.mean_train_x)
        np.save(self.output_path + "/std_x.npy", dataset.std_train_x)

        #note : in regression normalisation data for y should be redundant with x
        np.save(self.output_path + "/means_y.npy", dataset.mean_train_y)
        np.save(self.output_path + "/std_y.npy", dataset.std_train_y)


        sampler = samplers.PrioritizingSampler(dataset,self.multistep_test_number, alpha = self.alpha, beta = self.beta)

        # Session configuration : train only on CPU
        tfconfig = tf.ConfigProto(device_count = {'GPU': 0})

        # Saves training parameters
        train_logs = []
        test_logs = []
        test_logs_multi_step = []
        best_nn = 100000.0
        least_worse_nn = 10000.0
        start_time = datetime.datetime.now()

        #tensor def for shorter writung and better readibility :
        train_step = self.graph_mlp_loss.train_step
        x = self.graph_mlp_loss.x
        y = self.graph_mlp_loss.y
        keep_prob = self.graph_mlp_loss.keep_prob
        step = self.graph_mlp_loss.step
        y_ = self.graph_mlp_loss.y_
        acc_op = self.graph_mlp_loss.acc_op
        s_loss = self.graph_mlp_loss.s_loss
        weights = self.graph_mlp_loss.weights

        with self.graph_mlp_loss.g.as_default():

            with tf.Session(config = tfconfig) as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
                saver = tf.train.Saver()

                for i in range(self.iterations):
                    w_i = np.ones(self.bs)
                    batch_xs, batch_ys = sampler.next_train_batch_RNN(self.bs)
                    _ = sess.run(train_step, feed_dict = {x:batch_xs, y:batch_ys, keep_prob: 0.75, step: i, weights: w_i})
                    if i%10 == 0:
                        # Single step performance evaluation on train set
                        batch_x , batch_y = dataset.getTrainRNN(range(dataset.train_size))
                        acc = sess.run(acc_op, feed_dict = {x:batch_xs, y:batch_ys, keep_prob: 0.75, step: i})
                        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
                        train_logs.append([i, elapsed_time, acc])

                    if i%50 == 0:
                        # Single step performance evaluation on test set
                        batch_x , batch_y = dataset.getValRNN(range(1000))
                        acc = sess.run(acc_op,feed_dict = {x:batch_x, y:batch_y, keep_prob:1.0, step:i})
                        test_logs.append([i,acc])
                        worse = 0
                        avg = []
                        for j in range(self.multistep_test_number):
                            batch_x, batch_y = sampler.get_random_val_batch_RNN(self.test_window)

                            full = batch_x[0,:self.input_history * self.input_dim][np.newaxis,...]
                            predictions = []

                            for k in range(self.test_window - 1):
                                pred = sess.run(y_,feed_dict = {x: full, keep_prob:1.0, step: i})
                                predictions.append([pred[0,0],pred[0,1],pred[0,2],pred[0,3]])
                                cmd = batch_x[k + 1, self.output_dim:self.input_dim][np.newaxis,...]
                                new = np.concatenate((pred, cmd), axis=1)
                                old = full[0,:-self.input_dim][np.newaxis,...]
                                full = np.concatenate((new, old), axis=1)
                            predictions = np.array(predictions)
                            error_x1 = SK_MSE(predictions[:,0],batch_y[:-1,0])
                            error_x2 = SK_MSE(predictions[:,1],batch_y[:-1,1])
                            error_x3 = SK_MSE(predictions[:,2],batch_y[:-1,2])
                            error_x4 = SK_MSE(predictions[:,3],batch_y[:-1,3])
                            if np.mean([error_x1,error_x2,error_x3,error_x4]) > worse:
                                worse = np.mean([error_x1, error_x2, error_x3, error_x4])
                            avg.append(np.mean([error_x1, error_x2, error_x3, error_x4]))

                        avg = np.mean(avg)
                        if avg < best_nn:
                            best_nn = avg
                            NN_save_name = self.output_path + '/best_NN'
                            saver.save(sess, NN_save_name)
                        if worse < least_worse_nn:
                            least_worse_nn = worse
                            NN_save_name = self.output_path + '/least_worse_NN'
                            saver.save(sess, NN_save_name)
                        # Multi step performance evaluation on test set
                        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
                        test_logs_multi_step.append([i, elapsed_time, avg, worse])
                        if i%250 == 0:
                            print('Step: ' + str(i) + ', 1s acc:' + str(acc) + ', ' + str(self.test_window) + 's worse acc: ' + str(worse) + ', ' + str(self.test_window) + 's avg acc: ' + str(avg))

                # Update
                for k in range(self.per_iter):
                    ex, ey = sampler.DS.getTrainRNN(range(0,sampler.DS.train_size))
                    predictions = sess.run(s_loss,feed_dict = {x:ex, y:ey, keep_prob:1.0})
                    sampler.prioritize(predictions)

                    for i in range((k + 1) * self.iterations, (k + 2) * self.iterations):
                        batch_xs, batch_ys, w_i = sampler.next_train_batch_prioritized_multistep(self.bs)
                        score = sess.run(train_step, feed_dict = {x:batch_xs, y:batch_ys, keep_prob: 0.75, step: i, weights: w_i})
                        if i%10 == 0:
                            # Single step performance evaluation on train set
                            batch_x , batch_y = dataset.getTrainRNN(range(1000))
                            acc = sess.run(acc_op, feed_dict = {x:batch_xs, y:batch_ys, keep_prob: 0.75, step: i})
                            elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
                            train_logs.append([i, elapsed_time, acc])

                        if i%50 == 0:
                            # Single step performance evaluation on test set
                            batch_x , batch_y = dataset.getValRNN(range(dataset.val_size))
                            acc = sess.run(acc_op, feed_dict = {x:batch_x, y:batch_y, keep_prob:1.0, step:i})
                            test_logs.append([i, acc])
                            worse = 0
                            avg = []
                            for j in range(self.multistep_test_number):
                                batch_x, batch_y = sampler.get_random_val_batch_RNN(self.test_window)

                                full = batch_x[0,:self.input_history * self.input_dim][np.newaxis,...]
                                predictions = []

                                for k in range(self.test_window - 1):
                                    pred = sess.run(y_,feed_dict = {x: full, keep_prob:1.0, step: i})
                                    predictions.append([pred[0,0],pred[0,1],pred[0,2],pred[0,3]])
                                    cmd = batch_x[k + 1, self.output_dim:self.input_dim][np.newaxis,...]
                                    new = np.concatenate((pred, cmd), axis=1)
                                    old = full[0,:-self.input_dim][np.newaxis,...]
                                    full = np.concatenate((new, old), axis=1)
                                predictions = np.array(predictions)
                                error_x1 = SK_MSE(predictions[:,0],batch_y[:-1,0])
                                error_x2 = SK_MSE(predictions[:,1],batch_y[:-1,1])
                                error_x3 = SK_MSE(predictions[:,2],batch_y[:-1,2])
                                error_x4 = SK_MSE(predictions[:,3],batch_y[:-1,3])
                                if np.mean([error_x1,error_x2,error_x3,error_x4]) > worse:
                                    worse = np.mean([error_x1, error_x2, error_x3, error_x4])
                                avg.append(np.mean([error_x1, error_x2, error_x3, error_x4]))

                            avg = np.mean(avg)
                            if avg < best_nn:
                                best_nn = avg
                                NN_save_name = self.output_path + '/best_NN'
                                saver.save(sess, NN_save_name)
                            if worse < least_worse_nn:
                                least_worse_nn = worse
                                NN_save_name = self.output_path + '/least_worse_NN'
                                saver.save(sess, NN_save_name)
                            # Multi step performance evaluation on test set
                            elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
                            test_logs_multi_step.append([i, elapsed_time, avg, worse])
                            if i%250 == 0:
                                print('Step: ' + str(i) + ', 1s acc:' + str(acc) + ', ' + str(self.test_window) + 's worse acc: ' + str(worse) + ', ' + str(self.test_window) + 's avg acc: ' + str(avg))

                NN_save_name = self.output_path + '/final_NN'
                saver.save(sess, NN_save_name)

                print('#### TRAINING DONE ! ####')
                print('Best Accuracy reach for: ' + str(best_nn))
                print('Least Worse Accuracy reach for: ' + str(least_worse_nn))
                np.save(self.output_path + "/train_loss_log.npy", np.array(train_logs))
                np.save(self.output_path + "/test_single_step_loss_log.npy", np.array(test_logs))
                np.save(self.output_path + "/test_multi_step_loss_log.npy", np.array(test_logs_multi_step))


    def train_classic(self):
        dataset = hdf5_reader.ReaderBoatKingfisher(self.list_dataset_path,self.input_history, self.input_dim,self.output_forecast, self.output_dim, self.cmd_dim, self.split_test, self.split_val)#do all the test/valid splitting

        np.save(self.output_path + "/means.npy", dataset.mean)
        np.save(self.output_path + "/std.npy", dataset.std)

        #note : in regression normalisation data for y should be redundant with x


        sampler = samplers.UniformSampler(dataset,self.multistep_test_number)

        # Session configuration : train only on CPU
        tfconfig = tf.ConfigProto(device_count = {'GPU': 0})

        # Saves training parameters
        train_logs = []
        test_logs = []
        test_logs_multi_step = []
        best_1s_nn = 100000.0
        best_ms_nn = 100000.0
        least_worse_ms_nn = 100000.0
        start_time = datetime.datetime.now()

        with self.graph_mlp.g.as_default():
            with tf.Session(config = tfconfig) as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
                saver = tf.train.Saver()

                w_i = np.ones(self.bs)
                for i in range(self.iterations):
                    batch_xs, batch_ys = sampler.next_train_batch(self.bs)
                    _ = sess.run(self.graph_mlp.train_step, feed_dict = {self.graph_mlp.x:batch_xs, self.graph_mlp.y:batch_ys,  self.graph_mlp.weights:w_i, self.graph_mlp.keep_prob: 0.75, self.graph_mlp.step: i})

                    if i%10 == 0:
                        # Single step performance evaluation on train set
                        batch_x , batch_y = sampler.next_train_batch(1000)# arbitrary low value for efficiency, this session is just to get a rough idea of how the training is going ...
                        acc = sess.run(self.graph_mlp.acc_op, feed_dict = {self.graph_mlp.x:batch_xs, self.graph_mlp.y:batch_ys, self.graph_mlp.keep_prob: 0.75, self.graph_mlp.step: i})
                        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
                        train_logs.append([i] + [elapsed_time] + list(acc))
                        avg = np.mean(acc)
                        if  avg < best_1s_nn:
                            best_1s_nn = avg
                            NN_save_name = self.output_path + '/best_1S_NN'
                            saver.save(sess, NN_save_name)
                            

                    if i%50 == 0:
                        # Single step performance evaluation on test set
                        batch_x , batch_y = sampler.next_val_batch(dataset.val_size)
                        acc = sess.run(self.graph_mlp.acc_op, feed_dict = {self.graph_mlp.x:batch_x, self.graph_mlp.y:batch_y, self.graph_mlp.keep_prob:1.0, self.graph_mlp.step:i})
                        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
                        test_logs.append([i] + [elapsed_time]+list(acc))
                        worse = 0
                        avg = []
                        trajectories = []
                        GT = []
                        for j in range(self.multistep_test_number):
                            batch_x, batch_y = sampler.get_random_val_batch(self.test_window)
                            full = batch_x[0,:self.input_history * self.input_dim][np.newaxis,...]
                            predictions = []

                            for k in range(self.test_window - 1):
                                pred = sess.run(self.graph_mlp.y_,feed_dict = {self.graph_mlp.x: full, self.graph_mlp.keep_prob:1.0, self.graph_mlp.step: i})
                                predictions.append([pred[0,0],pred[0,1],pred[0,2]])
                                cmd = batch_x[k+1, -self.cmd_dim:][np.newaxis,...]
                                new = np.concatenate((pred, cmd), axis=1)
                                old = full[0,:-self.input_dim][np.newaxis,...]
                                full = np.concatenate((new,old), axis=1)
                            predictions = np.array(predictions)
                            error_x1 = SK_MSE(predictions[:,0],batch_y[:-1,0])
                            error_x2 = SK_MSE(predictions[:,1],batch_y[:-1,1])
                            error_x3 = SK_MSE(predictions[:,2],batch_y[:-1,2])
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


    def train_classic_ethz(self):
        dataset = hdf5_reader.ReaderDroneETHZ(self.list_dataset_path,self.input_history, self.input_dim,self.output_forecast, self.output_dim, self.cmd_dim, self.split_test, self.split_val)#do all the test/valid splitting

        np.save(self.output_path + "/means.npy", dataset.mean)
        np.save(self.output_path + "/std.npy", dataset.std)

        #note : in regression normalisation data for y should be redundant with x


        sampler = samplers.UniformSampler(dataset,self.multistep_test_number)

        # Session configuration : train only on CPU
        tfconfig = tf.ConfigProto(device_count = {'GPU': 0})

        # Saves training parameters
        train_logs = []
        test_logs = []
        test_logs_multi_step = []
        best_1s_nn = 100000.0
        best_ms_nn = 100000.0
        least_worse_ms_nn = 100000.0
        start_time = datetime.datetime.now()

        with self.graph_mlp.g.as_default():
            with tf.Session(config = tfconfig) as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
                saver = tf.train.Saver()

                w_i = np.ones(self.bs)
                for i in range(self.iterations):
                    batch_xs, batch_ys = sampler.next_train_batch(self.bs)
                    _ = sess.run(self.graph_mlp.train_step, feed_dict = {self.graph_mlp.x:batch_xs, self.graph_mlp.y:batch_ys,  self.graph_mlp.weights:w_i, self.graph_mlp.keep_prob: 0.75, self.graph_mlp.step: i})

                    if i%10 == 0:
                        # Single step performance evaluation on train set
                        batch_x , batch_y = sampler.next_train_batch(1000)# arbitrary low value for efficiency, this session is just to get a rough idea of how the training is going ...
                        acc = sess.run(self.graph_mlp.acc_op, feed_dict = {self.graph_mlp.x:batch_xs, self.graph_mlp.y:batch_ys, self.graph_mlp.keep_prob: 0.75, self.graph_mlp.step: i})
                        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
                        train_logs.append([i] + [elapsed_time] + list(acc))
                        avg = np.mean(acc)
                        if  avg < best_1s_nn:
                            best_1s_nn = avg
                            NN_save_name = self.output_path + '/best_1S_NN'
                            saver.save(sess, NN_save_name)

                    if i%50 == 0:
                        # Single step performance evaluation on test set
                        batch_x , batch_y = sampler.next_val_batch(dataset.val_size)
                        acc = sess.run(self.graph_mlp.acc_op, feed_dict = {self.graph_mlp.x:batch_x, self.graph_mlp.y:batch_y, self.graph_mlp.keep_prob:1.0, self.graph_mlp.step:i})
                        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
                        test_logs.append([i] + [elapsed_time]+list(acc))
                        worse = 0
                        avg = []
                        trajectories = []
                        GT = []
                        for j in range(self.multistep_test_number):
                            batch_x, batch_y = sampler.get_random_val_batch(self.test_window)
                            full = batch_x[0,:self.input_history * self.input_dim][np.newaxis,...]
                            predictions = []

                            for k in range(self.test_window - 1):
                                pred = sess.run(self.graph_mlp.y_,feed_dict = {self.graph_mlp.x: full, self.graph_mlp.keep_prob:1.0, self.graph_mlp.step: i})
                                predictions.append([pred[0,0],pred[0,1],pred[0,2],pred[0,3],pred[0,4],pred[0,5],pred[0,6]])
                                cmd = batch_x[k+1, -self.cmd_dim:][np.newaxis,...]
                                new = np.concatenate((pred, cmd), axis=1)
                                old = full[0,:-self.input_dim][np.newaxis,...]
                                full = np.concatenate((new,old), axis=1)
                            predictions = np.array(predictions)
                            error_x1 = SK_MSE(predictions[:,0],batch_y[:-1,0])
                            error_x2 = SK_MSE(predictions[:,1],batch_y[:-1,1])
                            error_x3 = SK_MSE(predictions[:,2],batch_y[:-1,2])
                            error_x4 = SK_MSE(predictions[:,3],batch_y[:-1,3])
                            error_x5 = SK_MSE(predictions[:,4],batch_y[:-1,4])
                            error_x6 = SK_MSE(predictions[:,5],batch_y[:-1,5])
                            error_x7 = SK_MSE(predictions[:,6],batch_y[:-1,6])
                            trajectories.append(predictions)
                            GT.append(batch_y[:-1,:7])
                            err = [error_x1, error_x2, error_x3, error_x4, error_x5, error_x6, error_x7]
                            if np.mean(err) > worse:
                                worse = np.mean(err)
                            avg.append(err)
                        err = np.mean(avg,axis=0)
                        avg = np.mean(avg)
                        if avg < best_ms_nn:
                            best_ms_nn = avg
                            trajectories = np.array(trajectories)
                            GT = np.array(GT)
                            np.save(self.output_path + '/best_trajectories_predicted',trajectories)
                            np.save(self.output_path + '/best_trajectories_groundtruth',GT)
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
    
    def train_classic_drone(self):
        dataset = hdf5_reader.ReaderDrone(self.list_dataset_path,self.input_history, self.input_dim,self.output_forecast, self.output_dim, self.cmd_dim, self.split_test, self.split_val)#do all the test/valid splitting

        np.save(self.output_path + "/means.npy", dataset.mean)
        np.save(self.output_path + "/std.npy", dataset.std)

        #note : in regression normalisation data for y should be redundant with x


        sampler = samplers.UniformSampler(dataset,self.multistep_test_number)

        # Session configuration : train only on CPU
        tfconfig = tf.ConfigProto(device_count = {'GPU': 0})

        # Saves training parameters
        train_logs = []
        test_logs = []
        test_logs_multi_step = []
        best_1s_nn = 100000.0
        best_ms_nn = 100000.0
        least_worse_ms_nn = 100000.0
        start_time = datetime.datetime.now()

        with self.graph_mlp.g.as_default():
            with tf.Session(config = tfconfig) as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
                saver = tf.train.Saver()

                w_i = np.ones(self.bs)
                for i in range(self.iterations):
                    batch_xs, batch_ys = sampler.next_train_batch(self.bs)
                    _ = sess.run(self.graph_mlp.train_step, feed_dict = {self.graph_mlp.x:batch_xs, self.graph_mlp.y:batch_ys,  self.graph_mlp.weights:w_i, self.graph_mlp.keep_prob: 0.75, self.graph_mlp.step: i})

                    if i%10 == 0:
                        # Single step performance evaluation on train set
                        batch_x , batch_y = sampler.next_train_batch(1000)# arbitrary low value for efficiency, this session is just to get a rough idea of how the training is going ...
                        acc = sess.run(self.graph_mlp.acc_op, feed_dict = {self.graph_mlp.x:batch_xs, self.graph_mlp.y:batch_ys, self.graph_mlp.keep_prob: 0.75, self.graph_mlp.step: i})
                        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
                        train_logs.append([i] + [elapsed_time] + list(acc))
                        avg = np.mean(acc)
                        if  avg < best_1s_nn:
                            best_1s_nn = avg
                            NN_save_name = self.output_path + '/best_1S_NN'
                            saver.save(sess, NN_save_name)

                    if i%50 == 0:
                        # Single step performance evaluation on test set
                        batch_x , batch_y = sampler.next_val_batch(dataset.val_size)
                        acc = sess.run(self.graph_mlp.acc_op, feed_dict = {self.graph_mlp.x:batch_x, self.graph_mlp.y:batch_y, self.graph_mlp.keep_prob:1.0, self.graph_mlp.step:i})
                        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
                        test_logs.append([i] + [elapsed_time]+list(acc))
                        worse = 0
                        avg = []
                        trajectories = []
                        GT = []
                        for j in range(self.multistep_test_number):
                            batch_x, batch_y = sampler.get_random_val_batch(self.test_window)
                            full = batch_x[0,:self.input_history * self.input_dim][np.newaxis,...]
                            predictions = []

                            for k in range(self.test_window - 1):
                                pred = sess.run(self.graph_mlp.y_,feed_dict = {self.graph_mlp.x: full, self.graph_mlp.keep_prob:1.0, self.graph_mlp.step: i})
                                predictions.append([pred[0,0],pred[0,1],pred[0,2],pred[0,3]])
                                cmd = batch_x[k+1, -self.cmd_dim:][np.newaxis,...]
                                new = np.concatenate((pred, cmd), axis=1)
                                old = full[0,:-self.input_dim][np.newaxis,...]
                                full = np.concatenate((new,old), axis=1)
                            predictions = np.array(predictions)
                            error_x1 = SK_MSE(predictions[:,0],batch_y[:-1,0])
                            error_x2 = SK_MSE(predictions[:,1],batch_y[:-1,1])
                            error_x3 = SK_MSE(predictions[:,2],batch_y[:-1,2])
                            error_x4 = SK_MSE(predictions[:,3],batch_y[:-1,3])
                            trajectories.append(predictions)
                            GT.append(batch_y[:-1,:4])
                            err = [error_x1, error_x2, error_x3, error_x4]
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

    def train_classic_5s(self):
        dataset = hdf5_reader.ReaderDrone(self.list_dataset_path, self.input_history, self.input_dim,self.output_forecast, self.output_dim, self.cmd_dim, self.split_test, self.split_val)#do all the test/valid splitting

        np.save(self.output_path + "/means_x.npy", dataset.mean_train_x)
        np.save(self.output_path + "/std_x.npy", dataset.std_train_x)

        #note : in regression normalisation data for y should be redundant with x
        np.save(self.output_path + "/means_y.npy", dataset.mean_train_y)
        np.save(self.output_path + "/std_y.npy", dataset.std_train_y)


        sampler = samplers.UniformSampler(dataset,self.multistep_test_number)

        # Session configuration : train only on CPU
        tfconfig = tf.ConfigProto(device_count = {'GPU': 0})

        # Saves training parameters
        train_logs = []
        test_logs = []
        test_logs_multi_step = []
        best_nn = 100000.0
        least_worse_nn = 10000.0
        start_time = datetime.datetime.now()

        #tensor def for shorter writung and better readibility :
        train_step = self.graph_mlp_loss.train_step
        x = self.graph_mlp_loss.x
        y = self.graph_mlp_loss.y
        keep_prob = self.graph_mlp_loss.keep_prob
        step = self.graph_mlp_loss.step
        weights = self.graph_mlp_loss.weights
        y_ = self.graph_mlp_loss.y_
        acc_op = self.graph_mlp_loss.acc_op

        with self.graph_mlp_loss.g.as_default():
            with tf.Session(config = tfconfig) as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
                saver = tf.train.Saver()

                w_i = np.ones(self.bs)
                for i in range(self.iterations):
                    batch_xs, batch_ys = sampler.next_train_batch_RNN(self.bs)
                    #print(batch_ys)
                    _ = sess.run(train_step, feed_dict = {x:batch_xs, y:batch_ys, keep_prob: 0.75, step: i, weights: w_i})

                    if i%10 == 0:
                        # Single step performance evaluation on train set
                        batch_x , batch_y = sampler.next_train_batch_RNN(1000)
                        acc = sess.run(acc_op, feed_dict = {x:batch_xs, y:batch_ys, keep_prob: 0.75, step: i})
                        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
                        train_logs.append([i,elapsed_time,acc])

                    if i%50 == 0:
                        # Single step performance evaluation on test set
                        batch_x , batch_y = sampler.next_val_batch_RNN(dataset.val_size)
                        acc = sess.run(acc_op,feed_dict = {x:batch_x, y:batch_y, keep_prob:1.0, step:i, weights: w_i})
                        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
                        test_logs.append([i]+[elapsed_time]+list(acc))
                        worse = 0
                        avg = []
                        for j in range(self.multistep_test_number):
                            batch_x, batch_y = sampler.get_random_val_batch_RNN(self.test_window)

                            full = batch_x[0,: self.input_history * self.input_dim][np.newaxis,...]
                            predictions = []

                            for k in range(self.test_window - 1):
                                pred = sess.run(y_,feed_dict = {x: full, keep_prob:1.0, step: i, weights: w_i})
                                predictions.append([pred[0,0],pred[0,1],pred[0,2],pred[0,3]])
                                cmd = batch_x[ k + 1, self.output_dim:self.input_dim][np.newaxis,...]
                                new = np.concatenate((pred, cmd), axis=1)
                                old = full[0,:-self.input_dim][np.newaxis,...]
                                full = np.concatenate((new, old), axis=1)
                            predictions = np.array(predictions)
                            error_x1 = SK_MSE(predictions[:,0],batch_y[:-1,0])
                            error_x2 = SK_MSE(predictions[:,1],batch_y[:-1,1])
                            error_x3 = SK_MSE(predictions[:,2],batch_y[:-1,2])
                            error_x4 = SK_MSE(predictions[:,3],batch_y[:-1,3])
                            if np.mean([error_x1,error_x2,error_x3,error_x4]) > worse:
                                worse = np.mean([error_x1, error_x2, error_x3, error_x4])
                            avg.append(np.mean([error_x1, error_x2, error_x3, error_x4]))

                        avg = np.mean(avg)
                        if avg < best_nn:
                            best_nn = avg
                            NN_save_name = self.output_path + '/best_NN'
                            saver.save(sess, NN_save_name)
                        if worse < least_worse_nn:
                            least_worse_nn = worse
                            NN_save_name = self.output_path + '/least_worse_NN'
                            saver.save(sess, NN_save_name)
                        # Multi step performance evaluation on test set
                        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
                        test_logs_multi_step.append([i, elapsed_time, avg, worse])
                        if i%250 == 0:
                            print('Step: ' + str(i) + ', 1s acc:' + str(acc) + ', ' + str(self.test_window) + 's worse acc: ' + str(worse) + ', ' + str(self.test_window) + 's avg acc: ' + str(avg))

                NN_save_name = self.output_path + '/final_NN'
                saver.save(sess, NN_save_name)
                print('#### TRAINING DONE ! ####')
                print('Best Accuracy reach for: ' + str(best_nn))
                print('Least Worse Accuracy reach for: ' + str(least_worse_nn))
                np.save(self.output_path + "/train_loss_log.npy", np.array(train_logs))
                np.save(self.output_path + "/test_single_step_loss_log.npy", np.array(test_logs))
                np.save(self.output_path + "/test_multi_step_loss_log.npy", np.array(test_logs_multi_step))

                np.save(self.output_path + "/result.npy",predictions)


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



