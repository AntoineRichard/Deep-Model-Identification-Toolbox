import os
import h5py as h5
import numpy as np

#TODO DOCU
#TODO Create trajectories on each file individually
#TODO Add continuity flag handling

class H5Reader:
    def __init__(self, settings):
        self.sts = settings

        self.std = None
        self.mean = None

        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
        self.val_x = None
        self.val_y = None
        
        self.test_traj_x = None
        self.test_traj_y = None
        self.test_val_x = None
        self.test_val_y = None
        
        self.load_data()

    def remove_ts(self, data):
        if self.ts_idx is not None:
           return data[:,[x for x in range(data.shape[1]) if x != self.sts.timestamp_idx]]
        else:
           return data

    def normalize(self, train_xy, test_xy, val_xy):
        self.mean = np.mean(train_xy, axis=0)
        self.std = np.std(train_xy, axis=0)
        train_xy = (train_xy - self.mean) / self.std
        test_xy = (test_xy - self.mean) / self.std
        val_xy = (val_xy - self.mean) / self.std
        return train_xy, test_xy, val_xy

    def load(self, root):
        files = os.listdir(root)
        datax = []
        datay = []
        for i in files:
            tmp = np.array(h5.File(os.path.join(root,i),'r')["train_X"])
            # Remove time-stamp if need be
            tmp = self.remove_ts(tmp)
            # split the input and targets
            tmp_x, tmp_y = self.split_input_output(tmp)
            # generates sequences for training
            tmp_x, tmp_y = self.sequence_generator(tmp_x, tmp_y)
            # append for concatenation
            data_x.append(tmpy)
            data_y.append(tmpx)
        numpy_data_x = np.concatenate((data_x), axis=0)
        numpy_data_y = np.concatenate((data_y), axis=0)
        return numpy_data

    def cross_validation_split(self, x, y):
        x_split = np.split(x, self.sts.folds)
        y_split = np.split(y, self.sts.folds)
        test_x = x_split[self.sts.test_idx]
        test_y = y_split[self.sts.test_idx]
        x = np.concatenate(x_split[[i for i in range(self.sts.folds) if i!=self.sts.test_idx]], axis=0)
        y = np.concatenate(y_split[[i for i in range(self.sts.folds) if i!=self.sts.test_idx]], axis=0)
        x_split = np.split(x, self.sts.folds)
        y_split = np.split(y, self.sts.folds)
        val_x = x_split[self.sts.val_idx]
        val_y = y_split[self.sts.val_idx]
        train_x = np.concatenate(x_split[[i for i in range(self.sts.folds) if i!=self.sts.val_idx]], axis=0)
        train_y = np.concatenate(y_split[[i for i in range(self.sts.folds) if i!=self.sts.val_idx]], axis=0)
        return train_x, train_y, test_x, test_y, val_x, val_y

    def ratio_based_split(self, x, y):
        raise('Not implemented')

    def split_input_output(self, xy):
        x = xy
        y = xy[:,:-self.sts.cmd_dim]
        return x, y
    
    def sequence_generator(self, x, y):
        nX = []
        nY = []
        # x is a sequence
        # y is the data-point right after the sequence
        for i in range(x.shape[0]-1-self.sts.sequence_length):
            nX.append(x[i:i+self.sts.sequence_length])
            nY.append(y[i+1+self.sts.sequence_length:i+self.sts.forecast+self.sts.sequence_length])
        nx = np.array(nX)
        ny = np.array(nY)
        return nx, ny
    
    def trajectory_generator(self, x, y):
        nX = []
        nY = []
        # x is a sequence
        # y is the data-point right after the sequence
        for i in range(x.shape[0]-1-self.sts.sequence_length-self.sts.trajectory_length):
            nX.append(x[i:i+self.sts.sequence_length+self.sts.trajectory_length])
            nY.append(y[i+self.sts.sequence_length+1:i+1+self.sts.sequence_length+self.trajectory_length])
        nx = np.array(nX)
        ny = np.array(nY)
        return nx, ny

    def load_data(self):
        # Load each dataset
        train_x, train_y = self.load(self.sts.train_dir)
        if (self.sts.test_dir is not None) and (self.sts.val_dir is not None):
            if self.sts.use_X_val:
                raise('Cannot use cross-validation with separated directory for training validation and testing.')
            else:
                test_xy = self.load(self.sts.test_dir)
                val_xy = self.load(self.sts.val_dir)
        elif self.sts.test_dir is None and self.sts.val_dir is not None:
            raise('Test root was not provided but validation root was, provide none or both.')
        elif self.sts.val_dir is None and self.sts.test_dir is not None:
            raise('Validation root was not provided but test root was, provide none or both.')
        elif self.sts.use_X_val:
            train_x, train_y, test_x, test_y, val_x, val_y = self.cross_validation_split(train_x, train_y)
        else:
            self.ratio_based_split(train_x, train_y)

        self.test_traj_x, self.test_traj_y = self.trajectory_generator(test_x, test_y)
        self.val_traj_x, self.val_traj_y = self.trajectory_generator(val_x, val_y)
        # normalize all dataset based on the train-set
        train_xy, test_xy, val_xy = self.normalize(train_x, train_y, test_x, test_y, val_x, val_y)
        # get sizes
        self.train_size = self.train_x.shape[0]
        self.test_size = self.test_x.shape[0]
        self.val_size = self.val_x.shape[0]
        self.test_traj_size = self.test_traj_x.shape[0]
        self.val_traj_size = self.val_traj_x.shape[0]
