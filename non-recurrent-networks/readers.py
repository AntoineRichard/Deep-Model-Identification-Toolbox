import os
import h5py as h5
import numpy as np
import csv
#TODO DOCU
#TODO Add continuity flag handling
#TODO Unify sequence_generator and trajectory generator

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
        if self.sts.timestamp_idx is not None:
           return data[:,[x for x in range(data.shape[1]) if x != self.sts.timestamp_idx]]
        else:
           return data

    def norm_state(self, x):
        return (x - self.mean) / self.std
    
    def norm_pred(self, y):
        return (y - self.mean[:-self.sts.cmd_dim]) / self.std[:-self.sts.cmd_dim]

    def normalize(self, train_x, train_y, test_x, test_y, val_x, val_y, test_traj_x, test_traj_y, val_traj_x, val_traj_y):
        self.mean = np.mean(np.mean(train_x, axis=0), axis=0)
        self.std = np.mean(np.std(train_x, axis=0), axis=0)

        self.train_x     = self.norm_state(train_x)
        self.test_x      = self.norm_state(test_x)     
        self.val_x       = self.norm_state(val_x)      
        self.test_traj_x = self.norm_state(test_traj_x)
        self.val_traj_x  = self.norm_state(val_traj_x) 

        self.train_y     = self.norm_pred(train_y)    
        self.test_y      = self.norm_pred(test_y)     
        self.val_y       = self.norm_pred(val_y)      
        self.test_traj_y = self.norm_pred(test_traj_y)
        self.val_traj_y  = self.norm_pred(val_traj_y)

    def read_file(self, path):
        return np.array(h5.File(path,'r')["train_X"])

    def load(self, root):
        files = os.listdir(root)
        data_x = []
        data_y = []
        data_traj_x = []
        data_traj_y = []
        for i in files:
            # Load file
            tmp = self.read_file(os.path.join(root,i))
            # Remove time-stamp if need be
            tmp = self.remove_ts(tmp)
            # split the input and targets
            tmp_x, tmp_y = self.split_input_output(tmp)
            # generate trajectories
            traj_x, traj_y = self.trajectory_generator(tmp_x, tmp_y)
            # generates sequences for training
            tmp_x, tmp_y = self.sequence_generator(tmp_x, tmp_y)
            # append for concatenation
            data_traj_x.append(traj_x)
            data_traj_y.append(traj_y)
            data_x.append(tmp_x)
            data_y.append(tmp_y)
        numpy_traj_x = np.concatenate((data_x), axis=0)
        numpy_traj_y = np.concatenate((data_y), axis=0)
        numpy_data_x = np.concatenate((data_x), axis=0)
        numpy_data_y = np.concatenate((data_y), axis=0)
        return numpy_data_x, numpy_data_y, traj_x, traj_y

    def split_var(self, x):
        x = x[:self.sts.folds*(int(x.shape[0]/self.sts.folds))]
        x_split = np.split(x, self.sts.folds)
        test_x = x_split[self.sts.test_idx]
        x = np.concatenate([x_split[i] for i in range(self.sts.folds) if i!=self.sts.test_idx], axis=0)
        x = x[:self.sts.folds*(int(x.shape[0]/self.sts.folds))]
        x_split = np.split(x, self.sts.folds)
        val_x = x_split[self.sts.val_idx]
        train_x = np.concatenate([x_split[i] for i in range(self.sts.folds) if i!=self.sts.val_idx], axis=0)
        return train_x, test_x, val_x

    def split_var_ratio(self, x):
        x, x_test = np.split(x,[int(-self.sts.test_ratio*x.shape[0])])
        x_train, x_val = np.split(x,[int(-self.sts.val_ratio*x.shape[0])])
        return x_train, x_test, x_val

    def cross_validation_split(self, x, y, x_traj, y_traj):
        train_x, test_x, val_x = self.split_var(x)
        train_y, test_y, val_y = self.split_var(y)
        _, test_traj_x, val_traj_x = self.split_var(x_traj)
        _, test_traj_y, val_traj_y = self.split_var(y_traj)
        return train_x, train_y, test_x, test_y, val_x, val_y, test_traj_x, test_traj_y, val_traj_x, val_traj_y

    def ratio_based_split(self, x, y, traj_x, traj_y):
        train_x, test_x, val_x = self.split_var_ratio(x)
        train_y, test_y, val_y = self.split_var_ratio(y)
        _, test_traj_x, val_traj_x = self.split_var_ratio(traj_x)
        _, test_traj_y, val_traj_y = self.split_var_ratio(traj_y)
        return train_x, train_y, test_x, test_y, val_x, val_y, test_traj_x, test_traj_y, val_traj_x, val_traj_y

    def split_input_output(self, xy):
        x = xy
        y = xy[:,:-self.sts.cmd_dim]
        return x, y
    
    def sequence_generator(self, x, y):
        nX = []
        nY = []
        # x is a sequence
        # y is the data-point or a sequence right after the sequence used as input
        for i in range(x.shape[0]-1-self.sts.sequence_length-self.sts.forecast):
            # First check continuity of the sequence if the flag is enabled
            if not (self.sts.continuity_idx is None):
                # 1 sequence is continuous 0 otherwise.
                vx = x[i:i+self.sts.sequence_length, self.sts.continuity_idx]
                vy = y[i+1+self.sts.sequence_length:i+1+self.sts.forecast+self.sts.sequence_length, self.sts.continuity_idx]
                # Check sequence is fine, if not skip sequence.
                if ((np.sum(vx) != vx.shape[0]) or (np.sum(vy) != vy.shape[0])):
                    continue
                else:
                    nX.append(x[i:i+self.sts.sequence_length,[xx for xx in range(x.shape[1])if xx!=self.sts.continuity_idx]])
                    nY.append(y[i+1+self.sts.sequence_length:i+1+self.sts.forecast+self.sts.sequence_length, [xx for xx in range(y.shape[1]) if xx!=self.sts.continuity_idx]])
            else:
                nX.append(x[i:i+self.sts.sequence_length])
                nY.append(y[i+1+self.sts.sequence_length:i+1+self.sts.forecast+self.sts.sequence_length])
        nx = np.array(nX)
        ny = np.array(nY)
        return nx, ny
    
    def trajectory_generator(self, x, y):
        nX = []
        nY = []
        # x is a sequence
        # y is a sequence right after the sequence used as input
        for i in range(x.shape[0]-1-self.sts.sequence_length-self.sts.trajectory_length):
            # First check continuity of the sequence if the flag is enabled
            if not (self.sts.continuity_idx is None):
                # 1 sequence is continuous 0 otherwise.
                vx = x[i:i+self.sts.sequence_length+self.sts.trajectory_length, self.sts.continuity_idx]
                vy = y[i:i+self.sts.sequence_length+self.sts.trajectory_length, self.sts.continuity_idx]
                # Check sequence is fine, if not skip sequence.
                if ((np.sum(vx) != vx.shape[0]) or (np.sum(vy) != vy.shape[0])):
                    continue
                else:
                    nX.append(x[i:i+self.sts.sequence_length+self.sts.trajectory_length,[xx for xx in range(x.shape[1])if xx!=self.sts.continuity_idx]])
                    nY.append(y[i+1+self.sts.sequence_length:i+1+self.sts.trajectory_length+self.sts.sequence_length, [xx for xx in range(y.shape[1]) if xx!=self.sts.continuity_idx]])
            else:
                nX.append(x[i:i+self.sts.sequence_length+self.sts.trajectory_length])
                nY.append(y[i+self.sts.sequence_length+1:i+1+self.sts.sequence_length+self.sts.trajectory_length])
        nx = np.array(nX)
        ny = np.array(nY)
        return nx, ny

    def load_data(self):
        # Load each dataset
        train_x, train_y, traj_x, traj_y = self.load(self.sts.train_dir)
        if (self.sts.test_dir is not None) and (self.sts.val_dir is not None):
            if self.sts.use_X_val:
                raise('Cannot use cross-validation with separated directory for training validation and testing.')
            else:
                test_x, test_y, test_traj_x, test_traj_y = self.load(self.sts.test_dir)
                val_x, val_y, val_traj_x, val_traj_y = self.load(self.sts.val_dir)
        elif self.sts.test_dir is None and self.sts.val_dir is not None:
            raise('Test root was not provided but validation root was, provide none or both.')
        elif self.sts.val_dir is None and self.sts.test_dir is not None:
            raise('Validation root was not provided but test root was, provide none or both.')
        elif self.sts.use_X_val:
            train_x, train_y, test_x, test_y, val_x, val_y, test_traj_x, test_traj_y, val_traj_x, val_traj_y  = self.cross_validation_split(train_x, train_y, traj_x, traj_y)
        else:
            train_x, train_y, test_x, test_y, val_x, val_y, test_traj_x, test_traj_y, val_traj_x, val_traj_y  = self.ratio_based_split(train_x, train_y, traj_x, traj_y)

        # normalize all dataset based on the train-set
        self.normalize(train_x, train_y, test_x, test_y, val_x, val_y, test_traj_x, test_traj_y, val_traj_x, val_traj_y)
        # get sizes
        self.train_size = self.train_x.shape[0]
        self.test_size = self.test_x.shape[0]
        self.val_size = self.val_x.shape[0]
        self.test_traj_size = self.test_traj_x.shape[0]
        self.val_traj_size = self.val_traj_x.shape[0]

class VolcaniReader_Forecast(H5Reader):
    def __init__(self, settings):
        super(VolcaniReader_Forecast, self).__init__(settings)
        
    def read_file(self, path):
        with open(path) as f:
            readCSV = csv.reader(f, delimiter=',')
            content = []
            for x in readCSV:
                content.append(x)
        
        header = content[0]
        data = content[1:]

        src_idx, tgt_idx = self.parse_header(header)
        return self.read_from_header(src_idx, tgt_idx, data)

    def parse_header(self, header):
        src_idx = [1000]*len(self.sts.source_header)
        tgt_idx = [1000]*len(self.sts.target_header)
        for i, name in enumerate(header):
            for j, src in enumerate(self.sts.source_header):
                if src == name:
                    src_idx[j] = i
            for j, tgt in enumerate(self.sts.target_header):
                if tgt == name:
                    tgt_idx[j] = i
    
        return src_idx, tgt_idx
    
    def read_from_header(self, src_idx, tgt_idx, data):
        new_data = []
        for x in data:
            src = [x[idx] for idx in src_idx]
            tgt = [x[idx] for idx in tgt_idx]
            cond = sum([1 for x in src+tgt if x]) == len(tgt+src)
            src = [float(x) if x else np.NaN for x in src]
            tgt = [float(x) if x else np.NaN for x in tgt]
            new_line = [cond*1]+tgt+src
            new_data.append(new_line)
        return np.array(new_data)

    def split_input_output(self, xy):
        x1 = np.expand_dims(xy[:,0],axis=1)
        x2 = xy[:,self.sts.output_dim+1:]
        x = np.concatenate((x1,x2),axis=1)
        y = xy[:,:self.sts.output_dim+1]
        return x, y
    
    def norm_state(self, x):
        x[:,:,0] = x[:,:,0]/24.0
        x[:,:,1:] = (x[:,:,1:] - self.mean[self.sts.output_dim+1:]) / self.std[self.sts.output_dim+1:]
        return x
    
    def norm_pred(self, y):
        return (y - self.mean[:self.sts.output_dim]) / self.std[:self.sts.output_dim]

    def normalize(self, train_x, train_y, test_x, test_y, val_x, val_y, test_traj_x, test_traj_y, val_traj_x, val_traj_y):
        mean_x = np.mean(np.mean(train_x, axis=0), axis=0)
        std_x = np.mean(np.std(train_x, axis=0), axis=0)
        #print(mean_x.shape)
        #print(std_x.shape)
        mean_y = np.mean(np.mean(train_y, axis=0), axis=0)
        std_y = np.mean(np.std(train_y, axis=0), axis=0)
        #print(mean_y.shape)
        #print(std_y.shape)
        self.mean = np.concatenate((mean_y,mean_x),axis=0)
        self.std = np.concatenate((std_y,std_x),axis=0)

        self.train_x     = self.norm_state(train_x)
        self.test_x      = self.norm_state(test_x)     
        self.val_x       = self.norm_state(val_x)      
        self.test_traj_x = self.norm_state(test_traj_x)
        self.val_traj_x  = self.norm_state(val_traj_x) 

        self.train_y     = self.norm_pred(train_y)    
        self.test_y      = self.norm_pred(test_y)     
        self.val_y       = self.norm_pred(val_y)      
        self.test_traj_y = self.norm_pred(test_traj_y)
        self.val_traj_y  = self.norm_pred(val_traj_y)
    
    def load_data(self):
        # Load each dataset
        train_x, train_y, traj_x, traj_y = self.load(self.sts.train_dir)
        if (self.sts.test_dir is not None) and (self.sts.val_dir is not None):
            if self.sts.use_X_val:
                raise('Cannot use cross-validation with separated directory for training validation and testing.')
            else:
                test_x, test_y, test_traj_x, test_traj_y = self.load(self.sts.test_dir)
                val_x, val_y, val_traj_x, val_traj_y = self.load(self.sts.val_dir)
        elif (self.sts.test_dir is None) and (self.sts.val_dir is not None):
            test_x, test_y, test_traj_x, test_traj_y = self.load(self.sts.val_dir)
            val_x, val_y, val_traj_x, val_traj_y = self.load(self.sts.val_dir)
        elif (self.sts.val_dir is None) and (self.sts.test_dir is not None):
            test_x, test_y, test_traj_x, test_traj_y = self.load(self.sts.test_dir)
            val_x, val_y, val_traj_x, val_traj_y = self.load(self.sts.test_dir)
        elif self.sts.use_X_val:
            train_x, train_y, test_x, test_y, val_x, val_y, test_traj_x, test_traj_y, val_traj_x, val_traj_y  = self.cross_validation_split(train_x, train_y, traj_x, traj_y)
        else:
            train_x, train_y, test_x, test_y, val_x, val_y, test_traj_x, test_traj_y, val_traj_x, val_traj_y  = self.ratio_based_split(train_x, train_y, traj_x, traj_y)

        # normalize all dataset based on the train-set
        self.normalize(train_x, train_y, test_x, test_y, val_x, val_y, test_traj_x, test_traj_y, val_traj_x, val_traj_y)
        # get sizes
        self.train_size = self.train_x.shape[0]
        self.test_size = self.test_x.shape[0]
        self.val_size = self.val_x.shape[0]
        self.test_traj_size = self.test_traj_x.shape[0]
        self.val_traj_size = self.val_traj_x.shape[0]
