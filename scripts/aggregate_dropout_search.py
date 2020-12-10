import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.transforms import offset_copy
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
    datasets = os.listdir(args.folder)
    datasets_path = [os.path.join(args.folder,x) for x in datasets]
    models = os.listdir(datasets_path[0])

    f1,ax1 = plt.subplots(nrows=len(datasets), ncols=len(models),figsize=(16,10)) 
    f2,ax2 = plt.subplots(nrows=len(datasets), ncols=len(models),figsize=(16,10))
    #plt.setp(ax1.flat, xlabel='Activations', ylabel='Learning Rates')
    #plt.setp(ax2.flat, xlabel='Activations', ylabel='Learning Rates')

    pad = 7
    
    for ax, col in zip(ax1[0], models):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')
    
    for ax, row in zip(ax1[:,0], datasets):
        ax.annotate(row, xy=(-0.5, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')
    
    for ax, col in zip(ax2[0], models):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')
    
    for ax, row in zip(ax2[:,0], datasets):
        ax.annotate(row, xy=(-0.5, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')
    
    f1.subplots_adjust(left=0.15, top=0.95)
    f2.subplots_adjust(left=0.15, top=0.95)

    for row_pos, ds in enumerate(zip(datasets,datasets_path)):
        models_path = [os.path.join(ds[1],x) for x in models]
        for col_pos, md in enumerate(zip(models,models_path)):
            # Read files
            param_list = os.listdir(md[1])
            params = [x.split('-') for x in param_list]
            print(params)
            param_1_list = np.array(params)[:,0].astype(np.float32)
            param_2_list = np.array(params)[:,1]
            param_1_list.sort()
            param_2_list.sort()
            param_1_list = np.unique(param_1_list)
            param_2_list = np.unique(param_2_list)
            bound_1 = np.shape(param_1_list)[0]
            bound_2 = np.shape(param_2_list)[0]
            ss_mat = np.ones((bound_1, bound_2))
            ms_mat = np.ones((bound_1, bound_2))
            for i, param_1 in enumerate(param_1_list):
                for j, param_2 in enumerate(param_2_list):
                    ss_acc, ms_acc, _, _ = read_statistics(os.path.join(md[1],str(param_1)+'-'+str(param_2)))
                    ss_mat[i,j] = ss_acc
                    ms_mat[i,j] = ms_acc
            # SingleStep Accuracy Map
            im = ax1[row_pos][col_pos].imshow(ss_mat,cmap='jet',origin='lower')
            #f1.colorbar(im, ax=ax1[row_pos][col_pos])
            ax1[row_pos][col_pos].set_ylabel('Learning Rates')
            ax1[row_pos][col_pos].set_yticks(list(range(bound_1)))
            ax1[row_pos][col_pos].set_yticklabels(param_1_list)
            ax1[row_pos][col_pos].set_xlabel('Drop Rates')
            ax1[row_pos][col_pos].set_xticks(list(range(bound_2)))
            ax1[row_pos][col_pos].set_xticklabels(param_2_list)
            plt.setp(ax1[row_pos][col_pos].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            # MultiStep Accuracy Map
            im = ax2[row_pos][col_pos].imshow(ms_mat,cmap='jet',origin='lower')
            #f2.colorbar(im, ax=ax2[row_pos][col_pos])
            ax2[row_pos][col_pos].set_ylabel('Learning Rates')
            ax2[row_pos][col_pos].set_yticks(list(range(bound_1)))
            ax2[row_pos][col_pos].set_yticklabels(param_1_list)
            ax1[row_pos][col_pos].set_ylabel('Drop Rates')
            ax2[row_pos][col_pos].set_xticks(list(range(bound_2)))
            ax2[row_pos][col_pos].set_xticklabels(param_2_list)
            plt.setp(ax2[row_pos][col_pos].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    f1.tight_layout()
    f2.tight_layout()
    f1.savefig(os.path.join(args.output,'ss_grid_search_image.png'),dpi=300)
    f2.savefig(os.path.join(args.output,'ms_grid_search_image.png'),dpi=300)

args = parse_args()
get_statistics(args)
