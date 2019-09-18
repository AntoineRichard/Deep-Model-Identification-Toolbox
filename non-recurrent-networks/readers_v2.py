import os
import h5py as h5
import numpy as np

#TODO DOCU
#TODO Create trajectories on each file individually
#TODO Add continuity flag handling
#TODO Add multi-step forecast support
#TODO Add cross-validation support (Homemade or Sklearn ?)
          # Custom made to select val/test/train area (indices)
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
        data = []
        for i in files:
            data.append(np.array(h5.File(os.path.join(root,i),'r')["train_X"]))
        numpy_data = np.concatenate((data), axis=0)
        return numpy_data

    def cross_validation_split(self):
        raise('Not implemented')

    def split_input_output(self, xy):
        x = xy
        y = xy[:,:-self.cmd_dim]
        return x, y
    
    def sequence_generator(self, x, y):
        nX = []
        nY = []
        # x is a sequence
        # y is the data-point right after the sequence
        for i in range(x.shape[0]-1-self.seq_len):
            nX.append(x[i:i+self.seq_len])
            nY.append(y[i+1+self.seq_len])
        nx = np.array(nX)
        ny = np.array(nY)
        return nx, ny
    
    def trajectory_generator(self, x, y):
        nX = []
        nY = []
        # x is a sequence
        # y is the data-point right after the sequence
        for i in range(x.shape[0]-1-self.seq_len-self.window):
            nX.append(x[i:i+self.seq_len+self.window])
            nY.append(y[i+self.seq_len+1:i+1+self.seq_len+self.window])
        nx = np.array(nX)
        ny = np.array(nY)
        return nx, ny

    def load_data(self):
        # Load each dataset
        train_xy = self.load(self.train_root)
        if (self.test_root is not None) and (self.val_root is not None):
            if self.sts.use_X_val:
                raise('Cannot use cross-validation with separated directory for training validation and testing.')
            else:
                test_xy = self.load(self.test_root)
                val_xy = self.load(self.val_root)
        elif self.test_root is None and self.val_root is not None:
            raise('Test root was not provided but validation root was, provide none or both.')
        elif self.val_root is None and self.test_root is not None:
            raise('Validation root was not provided but test root was, provide none or both.')
        else:
            train_xy, text_xy, val_xy = self.cross_validation_split()

        # Remove time-stamp if need be
        train_xy = self.remove_ts(train_xy)
        test_xy = self.remove_ts(test_xy)
        val_xy = self.remove_ts(val_xy)
        # normalize all dataset based on the train-set
        train_xy, test_xy, val_xy = self.normalize(train_xy, test_xy, val_xy)
        # split the input and targets
        train_x, train_y = self.split_input_output(train_xy)
        test_x, test_y = self.split_input_output(test_xy)
        val_x, val_y = self.split_input_output(val_xy)
        # generates sequences for training
        self.train_x, self.train_y = self.sequence_generator(train_x, train_y)
        self.test_x, self.test_y = self.sequence_generator(test_x, test_y)
        self.val_x, self.val_y = self.sequence_generator(val_x, val_y)
        self.test_traj_x, self.test_traj_y = self.trajectory_generator(test_x, test_y)
        self.val_traj_x, self.val_traj_y = self.trajectory_generator(val_x, val_y)
        # get sizes
        self.train_size = self.train_x.shape[0]
        self.test_size = self.test_x.shape[0]
        self.val_size = self.val_x.shape[0]
        self.test_traj_size = self.test_traj_x.shape[0]
        self.val_traj_size = self.val_traj_x.shape[0]
