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
    forward_time = []
    backward_time = []
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
    params = [x.split('_') for x in param_list]
    print(params)
    param_1_list = np.array(params)[:,0]#.astype(np.float32)
    param_2_list = np.array(params)[:,1]#.astype(np.int)
    #param_1_list.sort()
    #param_2_list.sort()
    param_1_list = np.unique(param_1_list)
    param_2_list = np.unique(param_2_list)
    print(param_1_list)
    print(param_2_list)
    bound_1 = np.shape(param_1_list)[0]
    bound_2 = np.shape(param_2_list)[0]
    ss_mat = np.ones((bound_1, bound_2))
    ms_mat = np.ones((bound_1, bound_2))
    for i, param_1 in enumerate(param_1_list):
        for j, param_2 in enumerate(param_2_list):
            ss_acc, ms_acc, _, _ = read_statistics(os.path.join(args.folder,str(param_1)+'_'+str(param_2)))
            ss_mat[i,j] = ss_acc
            ms_mat[i,j] = ms_acc
    # SingleStep Accuracy Map
    plt.imshow(ss_mat,cmap='jet',origin='lower')
    plt.colorbar()
    plt.title('SingleStep RMSE for different alpha/beta values')
    plt.ylabel('alpha')
    plt.xlabel('beta')
    plt.yticks(list(range(bound_1)),param_1_list.astype(np.float32)/10)
    plt.xticks(list(range(bound_2)),param_2_list.astype(np.float32)/10)
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(args.output,'ss_grid_search_image.png'))
    plt.cla()
    plt.clf()
    plt.close()
    # MultiStep Accuracy Map
    plt.imshow(ms_mat,cmap='jet',origin='lower')
    plt.colorbar()
    plt.title('MultiStep RMSE for different alpha/beta values')
    plt.ylabel('alpha')
    plt.xlabel('beta')
    plt.yticks(list(range(bound_1)),param_1_list.astype(np.float32)/10)
    plt.xticks(list(range(bound_2)),param_2_list.astype(np.float32)/10)
    plt.savefig(os.path.join(args.output,'ms_grid_search_image.png'))

args = parse_args()
get_statistics(args)
