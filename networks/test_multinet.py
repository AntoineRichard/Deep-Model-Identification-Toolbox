import numpy as np
import tensorflow as tf

import network_generator
from settings import Settings
import samplers
import readers
import models

sts = Settings()
DS = readers.H5Reader(sts)
SR = samplers.UniformSampler(DS, sts)
M_1 = network_generator.get_graph(sts)
M_2 = network_generator.get_graph(sts)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
#graph = tf.get_default_graph()
#for op in graph.get_operations():
#    print(op.name)
#
i = 0
prct, batch_xs, batch_ys = SR.sample_train_batch()
L1 = sess.run(M_1.s_loss, feed_dict = {M_1.x: batch_xs,
                                                         M_1.y: batch_ys,
                                                         M_1.weights: np.ones(sts.batch_size),
                                                         M_1.keep_prob: sts.dropout,
                                                         M_1.step: i,
                                                         M_1.is_training: True})
L2 = sess.run(M_2.s_loss, feed_dict = {M_2.x: batch_xs,
                                                         M_2.y: batch_ys,
                                                         M_2.weights: np.ones(sts.batch_size),
                                                         M_2.keep_prob: sts.dropout,
                                                         M_2.step: i,
                                                         M_2.is_training: True})
threshold = 0.9
args_from_L1 = np.argsort(L1)[:int(sts.batch_size*threshold)]
args_from_L2 = np.argsort(L2)[:int(sts.batch_size*threshold)]

_ = sess.run(M_1.train_step, feed_dict = { M_1.y: batch_ys[args_from_L2],
                                           M_1.x: batch_xs[args_from_L2],
                                           M_1.weights: np.ones(batch_ys[args_from_L2].shape[0]),
                                           M_1.step: i,
                                           M_1.is_training: True})
_ = sess.run(M_2.train_step, feed_dict = { M_2.y: batch_ys[args_from_L1],
                                           M_2.x: batch_xs[args_from_L1],
                                           M_2.weights: np.ones(batch_ys[args_from_L1].shape[0]),
                                           M_2.step: i,
                                           M_2.is_training: True})
sess.close()
