import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import argparse
import os
import shutil

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder',type=str, help='path to the train data folder to be used')
    parser.add_argument('--output',type=str, help='path to the train data folder to be used')
    parser.add_argument('--mode',type=str, help='type of gridsearch: per, grad, std')
    args = parser.parse_args()
    return args

def check_folder(path):
    if ~os.path.exists(path):
        raise ValueError('Specified path: '+path+' doesn\'t exist. Please enter a valid path.')

def read_stats(path):
    with open(os.path.join(path,'statistics.txt')) as fp: 
        Lines = fp.readlines() 
        for count, line in enumerate(Lines):
            if count == 0:
                parameters = int(line.split(':')[-1])
            elif count == 1:
                forward_time = float((line.split(':')[-1]).split(' ')[0])
            elif count == 2:
                backward_time = float((line.split(':')[-1]).split(' ')[0])
            elif count == 3:
                ss_accuracy = float(line.split(':')[-1])
            elif count == 4:
                ms_accuracy = float(line.split(':')[-1])
            elif count == 5:
                lw_accuracy = float(line.split(':')[-1])
            else:
                continue
    return parameters, forward_time, backward_time, ss_accuracy, ms_accuracy, lw_accuracy

def read_statistics(folder):
    forward_time = []
    backward_time = []
    ss_accuracy = []
    ms_accuracy = []
    
    runs = os.listdir(folder)
    for run in runs:
        run = os.path.join(folder,run)
        _, _, _, ss, ms, _ = read_stats(run)
        ss_accuracy.append(ss)
        ms_accuracy.append(ms)
    return np.mean(ss_accuracy), np.mean(ms_accuracy), np.std(ss_accuracy), np.std(ms_accuracy)#, np.min(ss_accuracy), np.max(ss_accuracy), np.min(ms_accuracy), np.max(ms_accurac)

def get_statistics(args):
    try:
        os.mkdir(args.output)
    except:
        pass

    if args.mode == 'per':
        # Read files
        param_list = os.listdir(args.folder)
        params = [x.split('_') for x in param_list]
        params_idx = np.array(params).astype(np.int64) - 1
        params = np.array(params).astype(np.float32)/10.
        bound = np.max(params_idx)+1
        ss_mat = np.ones((bound, bound))
        ms_mat = np.ones((bound, bound))
        for i, param in enumerate(param_list):
            ss_acc, ms_acc, _, _ = read_statistics(os.path.join(args.folder,param))
            ss_mat[params_idx[i][0],params_idx[i][1]] = ss_acc
            ms_mat[params_idx[i][0],params_idx[i][1]] = ms_acc
        # SingleStep Accuracy Map
        plt.imshow(ss_mat,cmap='jet',origin='lower')
        plt.colorbar()
        plt.title('SingleStep RMSE for different alpha/beta values')
        plt.xlabel('alpha')
        plt.ylabel('beta')
        plt.xticks(list(range(bound)),(np.array(list(range(bound)))+1)/10)
        plt.yticks(list(range(bound)),(np.array(list(range(bound)))+1)/10)
        plt.savefig(os.path.join(args.output,'ss_grid_search_image.png'))
        plt.cla()
        plt.clf()
        plt.close()
        # MultiStep Accuracy Map
        plt.imshow(ms_mat,cmap='jet',origin='lower')
        plt.colorbar()
        plt.title('MultiStep RMSE for different alpha/beta values')
        plt.xlabel('alpha')
        plt.ylabel('beta')
        plt.xticks(list(range(bound)),(np.array(list(range(bound)))+1)/10)
        plt.yticks(list(range(bound)),(np.array(list(range(bound)))+1)/10)
        plt.savefig(os.path.join(args.output,'ms_grid_search_image.png'))
        # Gets parameters
        # SingleStep models
        avg_ss_value = np.unravel_index(np.argmin(np.abs(ss_mat - np.mean(ss_mat)), axis=None), ss_mat.shape)
        max_ss_value = np.unravel_index(np.argmax(ss_mat, axis=None), ss_mat.shape)
        min_ss_value = np.unravel_index(np.argmin(ss_mat, axis=None), ss_mat.shape)
        avg_ss_param = '0'+str(avg_ss_value[0]+1)+'_0'+str(avg_ss_value[1]+1)
        worst_ss_param = '0'+str(max_ss_value[0]+1)+'_0'+str(max_ss_value[1]+1)
        best_ss_param = '0'+str(min_ss_value[0]+1)+'_0'+str(min_ss_value[1]+1)
        # MultiStep models
        avg_ms_value = np.unravel_index(np.argmin(np.abs(ms_mat - np.mean(ms_mat)), axis=None), ms_mat.shape)
        max_ms_value = np.unravel_index(np.argmax(ms_mat, axis=None), ms_mat.shape)
        min_ms_value = np.unravel_index(np.argmin(ms_mat, axis=None), ms_mat.shape)
        avg_ms_param = '0'+str(avg_ms_value[0]+1)+'_0'+str(avg_ms_value[1]+1)
        worst_ms_param = '0'+str(max_ms_value[0]+1)+'_0'+str(max_ms_value[1]+1)
        best_ms_param = '0'+str(min_ms_value[0]+1)+'_0'+str(min_ms_value[1]+1)

    elif args.mode == 'grad':
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
        plt.title('SingleStep RMSE for different superbatch sizes')
        plt.xlabel('superbatch size')
        plt.ylabel('rmse')
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
        plt.title('SingleStep RMSE for different superbatch sizes')
        plt.xlabel('superbatch size')
        plt.ylabel('rmse')
        plt.legend()
        plt.savefig(os.path.join(args.output,'ms_grid_search_plot.png'))
        # Gets parameters
        # SingleStep models
        avg_ss_value = np.unravel_index(np.argmin(np.abs(ss_mat - np.mean(ss_mat)), axis=None), ss_mat.shape)
        max_ss_value = np.unravel_index(np.argmax(ss_mat, axis=None), ss_mat.shape)
        min_ss_value = np.unravel_index(np.argmin(ss_mat, axis=None), ss_mat.shape)
        avg_ss_param = str(params[avg_ss_value])
        worst_ss_param = str(params[max_ss_value])
        best_ss_param = str(params[min_ss_value])
        # MultiStep models
        avg_ms_value = np.unravel_index(np.argmin(np.abs(ms_mat - np.mean(ms_mat)), axis=None), ms_mat.shape)
        max_ms_value = np.unravel_index(np.argmax(ms_mat, axis=None), ms_mat.shape)
        min_ms_value = np.unravel_index(np.argmin(ms_mat, axis=None), ms_mat.shape)
        avg_ms_param = str(params[avg_ms_value])
        worst_ms_param = str(params[max_ms_value])
        best_ms_param = str(params[min_ms_value])
    elif args.mode == 'std':
        pass
    else:
        raise ValueError('Specified mode not supported: '+args.mode+'. Currently supported modes are: \"per\", \"grad\", and \"std\"')

args = parse_args()
get_statistics(args)
