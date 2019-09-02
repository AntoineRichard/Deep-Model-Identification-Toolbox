import argparse
import numpy as np
import tensorflow as tf


def parse_args(parser):
    parser.add_argument('-d','--dataset',type=str, default='..', help='path to the data file to be used')
    parser.add_argument('--output', type=str, default='./result', help='path to the folder to save file to')
    parser.add_argument('--split_test', type=int, default='0', help='split test/train+val set for  cross-validation')
    parser.add_argument('--split_val', type=int, default='0', help='split train/val set for  cross-validation')
    parser.add_argument('--batch_size', type=int, default='16', help='size of the batch')
    parser.add_argument('--super_batch_size', type=int, default='512', help='size of the super batch')
    parser.add_argument('--iterations', type=int, default='10000', help='number of iterations')
    parser.add_argument('--used_points', type=int, default='12', help='number of points used to predict the outputs')
    parser.add_argument('--pred_points', type=int, default='1', help='number of point to predict')
    parser.add_argument('--frequency', type=int, default='1', help='sampling frequency')
    parser.add_argument('--test_window', type=int, default='30', help='multi step test window size')
    parser.add_argument('--test_iter', type=int, default='5', help='multi step test iteration')
    parser.add_argument('--alpha', type=float, default='0.4', help='PER param alpha')
    parser.add_argument('--beta', type=float, default='0.5', help='PER param beta')
    parser.add_argument('--experience_replay', type=int, default='4', help='PER replay times')

    
    args = parser.parse_args()
    
    return args

def accuracy(estimation):
    accuracy = tf.reduce_mean(tf.cast(estimation, tf.float32))
    return accuracy

def train_fn(loss, learning_rate):
    with tf.variable_scope('train'):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_step

def get_random_test_batch(dataset,args) :
    return dataset.getWindowTest(args)

def get_random_test_batch_RNN(dataset,args) : 
    return dataset.getWindowTestRNN(args)

def get_random_val_batch(dataset,args) :
    return dataset.getWindowVal(args)

def get_random_val_batch_RNN(dataset,args) :
    return dataset.getWindowValRNN(args)

