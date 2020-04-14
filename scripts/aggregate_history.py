import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import argparse
import os
import shutil
import ast

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder',type=str, help='path to the train data folder to be used')
    parser.add_argument('--output',type=str, help='path to the train data folder to be used')
    args = parser.parse_args()
    return args

def check_folder(path):
    if ~os.path.exists(path):
        raise ValueError('Specified path: '+path+' doesn\'t exist. Please enter a valid path.')

def read_stats(path):
    with open(os.path.join(path,'statistics.txt')) as fp: 
        Lines = fp.readlines() 
        for count, line in enumerate(Lines):
            if count == 6:
                tmp = [i.strip() for i in line.split(':')[-1].split(' ') if i]
                tmp[0] = tmp[0][1:]
                tmp[-1] = tmp[-1][:-1]
                tmp = [i for i in tmp if i]
                test_ss_accuracy = np.array([float(n) for n in tmp])
            elif count == 7:
                tmp = [i.strip() for i in line.split(':')[-1].split(' ') if i]
                tmp[0] = tmp[0][1:]
                tmp[-1] = tmp[-1][:-1]
                tmp = [i for i in tmp if i]
                test_ms_accuracy = np.array([float(n) for n in tmp])
            elif count == 8:
                tmp = [i.strip() for i in line.split(':')[-1].split(' ') if i]
                tmp[0] = tmp[0][1:]
                tmp[-1] = tmp[-1][:-1]
                tmp = [i for i in tmp if i]
                test_ms_std = np.array([float(n) for n in tmp])
            else:
                continue
    return test_ss_accuracy, test_ms_accuracy, test_ms_std

def read_statistics(folder):
    ss_accuracy = []
    ms_accuracy = []
    ms_std_dev = []
    
    runs = os.listdir(folder)
    for run in runs:
        run = os.path.join(folder,run)
        ss, ms, ms_std = read_stats(run)
        ss_accuracy.append(ss)
        ms_accuracy.append(ms)
        ms_std_dev.append(ms_std)
    return np.mean(ss_accuracy), np.mean(ms_accuracy), np.mean(np.std(ss_accuracy,axis=0)), np.mean(ms_std)

def get_statistics(args):
    try:
        os.mkdir(args.output)
    except:
        pass
    # Read files
    param_list = os.listdir(args.folder)
    params = np.array(param_list).astype(np.int64)
    params.sort()
    bound = params.shape[0]
    ss_mat = np.ones((bound))
    ms_mat = np.ones((bound))
    sss_mat = np.ones((bound))
    mss_mat = np.ones((bound))
    for i, param in enumerate(params):
        ss_acc, ms_acc, ss_std, ms_std = read_statistics(os.path.join(args.folder,str(param)))
        ss_mat[i] = ss_acc
        ms_mat[i] = ms_acc
        sss_mat[i] = ss_std
        mss_mat[i] = ms_std
    # SingleStep Accuracy Histogram
    y1 = ss_mat - sss_mat
    y2 = ss_mat + sss_mat
    plt.plot(params, y1, params, y2, color='orange')
    plt.fill_between(params, y1, y2, where=y2 >= y1, facecolor='orange', alpha=0.5, interpolate=True, label='std_dev')
    plt.plot(params, ss_mat, color='C0', label='average')
    plt.title('Is longer better ? MLP\'s singlestep predictions')
    plt.xlabel('history length')
    plt.ylabel('RMSE')
    plt.legend()
    plt.savefig(os.path.join(args.output,'ss_grid_search_plot.png'))
    plt.cla()
    plt.clf()
    plt.close()
    # SingleStep Accuracy Histogram
    y1 = ms_mat - mss_mat
    y2 = ms_mat + mss_mat
    plt.plot(params, y1, params, y2, color='orange')
    plt.fill_between(params, y1, y2, where=y2 >= y1, facecolor='orange', alpha=0.5, interpolate=True, label='std_dev')
    plt.plot(params, ms_mat, color='C0', label='average')
    plt.title('Is longer better ? MLP\'s multistep predictions')
    plt.xlabel('history length')
    plt.ylabel('RMSE')
    plt.ylim([0,np.mean(y1)*3])
    plt.legend()
    plt.savefig(os.path.join(args.output,'ms_grid_search_plot.png'))

args = parse_args()
get_statistics(args)
