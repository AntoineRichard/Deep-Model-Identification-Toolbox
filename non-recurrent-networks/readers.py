import os
import h5py as h5
import numpy as np
import csv
#TODO DOCU
#TODO Unify sequence_generator and trajectory generator

class H5Reader:
    """
    A reader object made to be compatible with our datasets formating:
    Time-stamp, Continuity-bit, States (1 or more), Commands (1 or more)
    This object prepares the data (Nomalizes, removes time-stamps, and
    continuity idx), and split it using the settings chosen by the user.
    Since we aim to apply our models to MPC, our models will be
    evaluated on multistep predictions (build on their predictions) 
    hence this reader also generates trajectories.
    """
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
        """
        Removes the timestamp if the timestamps are present in the data. (To set
        the index of the time-stamps see the settings object).
        Input:
            data: The raw data [N x Row_size]
        Output:
            data: The data without time stamp [N x Row_size -1]
        """
        if self.sts.timestamp_idx is not None:
           return data[:,[x for x in range(data.shape[1]) if x != self.sts.timestamp_idx]]
        else:
           return data

    def norm_state(self, x):
        """
        Normalizes the states. The normalization is done using the formula
        (X-MEAN)/STD, where X is the data, MEAN the average value for each
        variables and STD the standard deviation for each variables.
        Input:
            x: a vector of predictions [N x Sequence_size x Input_size]
        Output:
            x: the same vector but normalized [N x Sequence_size x Input_size]
        """
        return (x - self.mean) / self.std
    
    def norm_pred(self, y):
        """
        Normalizes the predictions or target: the ouput of the network
        The normalization is done using the formula (Y-MEAN)/STD, where Y is the data,
        MEAN the average value for each variables and STD the standard deviation 
        for each variables.
        Input:
            y: a vector of predictions [N x Forecast_size xOutput_size]
        Output:
            y: the same vector but normalized [N x Forecast_size x Ouput_size]
        """
        return (y - self.mean[:-self.sts.cmd_dim]) / self.std[:-self.sts.cmd_dim]

    def normalize(self, train_x, train_y, test_x, test_y, val_x, val_y, test_traj_x, test_traj_y, val_traj_x, val_traj_y):
        """
        Computes the means of the dataset and normalizes the output
        Input:
            train_x : input training values [Train_size x Sequence_size x Input_size]
            train_y : output training values [Train_size x Forecast_size x Output_size]
            test_x  : input test values [Test_size x Sequence_size x Input_size]
            test_y  : output test values [Test_size x Forecast_size x Output_size]
            val_x   : input validation values [Val_size x Sequence_size x Input_size]
            val_y   : output validation values [Val_size x Forecast_size x Output_size]
            test_traj_x : input test trajectories values [Test_size x Trajectory_size + Sequence_size x Input_size]
            test_traj_y : output test trajectories values [Test_size x Trajectory_size x Output_size] 
            val_traj_x  : input val trajectories values [Val_size x Trajectory_size + Sequence_size x Input_size]
            val_traj_y  : output val trajectories values [Val_size x Trajectory_size x Output_size]
        Output:
            None (saved in class)
        """
        # Compute normalization variables
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
        """
        Reads the HDF5 file and casts it into a numpy array.
        """
        return np.array(h5.File(path,'r')["train_X"])

    def load(self, root):
        """
        Loads the whole of the training set
        """
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
        """
        Splits the data into a K-fold Cross-Validation form where the number of fold is 
        ajustable and position of the train the test or the validation set can be changed
        using indices. The numbers of folds and the index of each set can be set in the
        settings object.
        Input:
            x : the data in the form [N x M] or [N x M x O]
        Output:
            train_x : the training data in the form [N x M] or [N x M x O]
            test_x  : the test data in the form [N x M] or [N x M x O]
            val_x   : the validation data in the form [N x M] or [N x M x O]
        """
        # Cuts the data into folds
        x = x[:self.sts.folds*(int(x.shape[0]/self.sts.folds))]
        x_split = np.split(x, self.sts.folds)
        # Pick the test
        test_x = x_split[self.sts.test_idx]
        # Fuse and split
        x = np.concatenate([x_split[i] for i in range(self.sts.folds) if i!=self.sts.test_idx], axis=0)
        x = x[:self.sts.folds*(int(x.shape[0]/self.sts.folds))]
        x_split = np.split(x, self.sts.folds)
        # Pick val
        val_x = x_split[self.sts.val_idx]
        # Fuse train
        train_x = np.concatenate([x_split[i] for i in range(self.sts.folds) if i!=self.sts.val_idx], axis=0)
        return train_x, test_x, val_x

    def split_var_ratio(self, x):
        """
        Splits the data into [Train | Val | Test] where the size of the test is a percentage of the whole data
        defined by the setting test_ratio (in the settings object), the size of the val is a percentage of the
        remaining data defined by val_ratio (in the settings object) and the size of the training set is the 
        rest of the data.
        Input:
            x : The whole data [N x M x O]
        Output:
            x_train : The training set [N x M x O]
            x_test  : The test set [N x M x O]
            x_val   : The val set [N x M x O]
        """
        x, x_test = np.split(x,[int(-self.sts.test_ratio*x.shape[0])])
        x_train, x_val = np.split(x,[int(-self.sts.val_ratio*x.shape[0])])
        return x_train, x_test, x_val

    def cross_validation_split(self, x, y, x_traj, y_traj):
        """
        Calls the spliting methode related to the K-Fold Cross Validation and applies it to
        the x, y, x_traj and y_traj
        Input:
            x : Generated sequence in the form [N x Sequence_size x Input_dim]
            y : Generated sequence in the form [N x Forecast_size x Output_dim]
            x_traj : Generated trajectories in the form [N x Trajectory_size + Sequence_size x Input_size]
            y_traj : Generated trajectories in the from [N x Trajectory_size x Output_size]
        Output:
            train_x : The input training sequences [N x Sequence_size x Input_dim]
            train_y : The output training sequences [N x Forecast_size x Output_dim]
            test_x  : The input test sequences [N x Sequence_size x Input_dim]
            test_y  : The output test sequences [N x Forecast_size x Output_dim]
            val_x   : The input validation sequences [N x Sequence_size x Input_dim]
            val_y   : The output validation sequences [N x Forecast_size x Output_dim]
            test_traj_x : The input test trajectories [N x Trajectory_size + Sequence_size x Input_dim]
            test_traj_y : The output test trajectories [N x Trajectory_size x Output_dim]
            val_traj_x  : The input validation trajectories [N x Trajectory_size + Sequence_size x Input_dim]
            val_traj_y  : The output validation trajectories  [N x Trajectory_size x Output_dim]
        """
        train_x, test_x, val_x = self.split_var(x)
        train_y, test_y, val_y = self.split_var(y)
        _, test_traj_x, val_traj_x = self.split_var(x_traj)
        _, test_traj_y, val_traj_y = self.split_var(y_traj)
        return train_x, train_y, test_x, test_y, val_x, val_y, test_traj_x, test_traj_y, val_traj_x, val_traj_y

    def ratio_based_split(self, x, y, traj_x, traj_y):
        """
        Calls the spliting methode related to the ratio based split and applies it to
        the x, y, x_traj and y_traj
        Input:
            x : Generated sequence in the form [N x Sequence_size x Input_dim]
            y : Generated sequence in the form [N x Forecast_size x Output_dim]
            x_traj : Generated trajectories in the form [N x Trajectory_size + Sequence_size x Input_size]
            y_traj : Generated trajectories in the from [N x Trajectory_size x Output_size]
        Output:
            train_x : The input training sequences [N x Sequence_size x Input_dim]
            train_y : The output training sequences [N x Forecast_size x Output_dim]
            test_x  : The input test sequences [N x Sequence_size x Input_dim]
            test_y  : The output test sequences [N x Forecast_size x Output_dim]
            val_x   : The input validation sequences [N x Sequence_size x Input_dim]
            val_y   : The output validation sequences [N x Forecast_size x Output_dim]
            test_traj_x : The input test trajectories [N x Trajectory_size + Sequence_size x Input_dim]
            test_traj_y : The output test trajectories [N x Trajectory_size x Output_dim]
            val_traj_x  : The input validation trajectories [N x Trajectory_size + Sequence_size x Input_dim]
            val_traj_y  : The output validation trajectories  [N x Trajectory_size x Output_dim]
        """
        train_x, test_x, val_x = self.split_var_ratio(x)
        train_y, test_y, val_y = self.split_var_ratio(y)
        _, test_traj_x, val_traj_x = self.split_var_ratio(traj_x)
        _, test_traj_y, val_traj_y = self.split_var_ratio(traj_y)
        return train_x, train_y, test_x, test_y, val_x, val_y, test_traj_x, test_traj_y, val_traj_x, val_traj_y

    def split_input_output(self, xy):
        """
        Separates the input and outputs
        Input:
            xy : full dataset in the following format: [N x M] where the timestamp have been removed.
            The data as to be in the form [Input | Output] or [Input | Output + Continuity_idx]
        Output:
            x : The inputs in the form [N x Input_dim] or [N x 1 + Input_dim] 
            y : The outputs in the form [N x Output_dim] or [N x 1 + Output_dim]
        """
        x = xy
        if not (self.sts.continuity_idx is None):
            value_idx = [xx for xx in range(x.shape[1])if xx!=self.sts.continuity_idx]
            y = xy[:,value_idx]
            y = y[:,:self.sts.output_dim]
            continuity_flag = xy[:, self.sts.continuity_idx]
            print(y.shape)
            return x, np.hstack((np.expand_dims(continuity_flag,-1),y))
        else:
            y = xy[:,:self.sts.output_dim]
            return x, y
    
    def sequence_generator(self, x, y):
        """
        Generates a sequence of data to be fed to the network.
        Input:
           x: The inputs in the form [N x Input_dim] or [N x 1 + Input_dim] 
           y: The outputs in the form [N x Output_dim] or [N x 1 + Output_dim]
        Output:
           nx: Generated sequence in the form [N x Sequence_size x Input_dim]
           ny: Generated sequence in the form [N x Forecast_size x Output_dim]
        """
        nX = []
        nY = []
        
        # Stores the indices of all the variables but the continuity index
        value_x_idx = [xx for xx in range(x.shape[1])if xx!=self.sts.continuity_idx]
        value_y_idx = [xx for xx in range(y.shape[1])if xx!=self.sts.continuity_idx]
        
        # x is a sequence, y is the data-point or a sequence right after the sequence used as input
        for i in range(x.shape[0]-1-self.sts.sequence_length-self.sts.forecast):
            # First check continuity of the sequence if the flag is enabled
            if not (self.sts.continuity_idx is None):
                # flag > 1 => manual control
                vx = x[i:i+self.sts.sequence_length, self.sts.continuity_idx]
                vy = y[i+1+self.sts.sequence_length:i+1+self.sts.forecast+self.sts.sequence_length, self.sts.continuity_idx]
                # Check sequence is fine, if not skip sequence.
                # if one of the flag is superior to one, then skip the seq
                if ((np.max(vx) > 1) or (np.max(vy) > 1)):
                    continue
                else:
                    nX.append(x[i:i+self.sts.sequence_length, value_x_idx])
                    nY.append(y[i+1+self.sts.sequence_length:i+1+self.sts.forecast+self.sts.sequence_length, value_y_idx])
            else:
                nX.append(x[i:i+self.sts.sequence_length])
                nY.append(y[i+1+self.sts.sequence_length:i+1+self.sts.forecast+self.sts.sequence_length])
        nx = np.array(nX)
        ny = np.array(nY)
        return nx, ny
    
    def trajectory_generator(self, x, y):
        """
        Generates a sequence of data to be fed to the network.
        Input:
           x: The inputs in the form [N x Input_dim] or [N x 1 + Input_dim] 
           y: The outputs in the form [N x Output_dim] or [N x 1 + Output_dim]
        Output:
           nx: Generated sequence in the form [N x Sequence_size x Input_dim]
           ny: Generated sequence in the form [N x Forecast_size x Output_dim]
        """
        nX = []
        nY = []
        
        # Stores the indices of all the variables but the continuity index
        value_x_idx = [xx for xx in range(x.shape[1])if xx!=self.sts.continuity_idx]
        value_y_idx = [xx for xx in range(y.shape[1])if xx!=self.sts.continuity_idx]
         
        # x is a sequence, y is a sequence right after the sequence used as input
        for i in range(x.shape[0]-1-self.sts.sequence_length-self.sts.trajectory_length):
            # First check continuity of the sequence if the flag is enabled
            if not (self.sts.continuity_idx is None):
                # 1 sequence is continuous 0 otherwise.
                vx = x[i:i+self.sts.sequence_length+self.sts.trajectory_length, self.sts.continuity_idx]
                vy = y[i+1:i+1+self.sts.sequence_length+self.sts.trajectory_length, self.sts.continuity_idx]
                # Check sequence is fine, if not skip sequence.
                if ((np.max(vx) > 1) or (np.max(vy) > 1)):
                    continue
                else:
                    nX.append(x[i:i+self.sts.sequence_length+self.sts.trajectory_length,value_x_idx])
                    nY.append(y[i+1+self.sts.sequence_length:i+1+self.sts.trajectory_length+self.sts.sequence_length, value_y_idx])
            else:
                nX.append(x[i:i+self.sts.sequence_length+self.sts.trajectory_length])
                nY.append(y[i+self.sts.sequence_length+1:i+1+self.sts.sequence_length+self.sts.trajectory_length])
        nx = np.array(nX)
        ny = np.array(nY)
        return nx, ny

    def load_data(self):
        """
        Build the dataset and splits it based on user input
        """
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

class H5Reader_Seq2Seq(H5Reader):
    def __init__(self, settings):
        super(H5Reader_Seq2Seq, self).__init__(settings)

    def sequence_generator(self, x, y):
        """
        Generates a sequence of data to be fed to the network in a Seq2Seq fashion.
        Input:
           x: The inputs in the form [N x Input_dim] or [N x 1 + Input_dim] 
           y: The outputs in the form [N x Output_dim] or [N x 1 + Output_dim]
        Output:
           nx: Generated sequence in the form [N x Sequence_size x Input_dim]
           ny: Generated sequence in the form [N x Sequence_size x Output_dim]
        """
        nX = []
        nY = []
        
        # Stores the indices of all the variables but the continuity index
        value_x_idx = [xx for xx in range(x.shape[1])if xx!=self.sts.continuity_idx]
        value_y_idx = [xx for xx in range(y.shape[1])if xx!=self.sts.continuity_idx]
        
        # x is a sequence, y is the data-point or a sequence right after the sequence used as input
        for i in range(x.shape[0]-1-self.sts.sequence_length-self.sts.forecast):
            # First check continuity of the sequence if the flag is enabled
            if not (self.sts.continuity_idx is None):
                # 1 sequence is continuous 0 otherwise.
                vx = x[i:i+self.sts.sequence_length, self.sts.continuity_idx]
                vy = y[i+1+self.sts.sequence_length:i+1+self.sts.forecast+self.sts.sequence_length, self.sts.continuity_idx]
                # Check sequence is fine, if not skip sequence.
                if ((np.max(vx) > 1) or (np.max(vy) > 1)):
                    continue
                else:
                    nX.append(x[i:i+self.sts.sequence_length, value_x_idx])
                    nY.append(y[i+1:i+1+self.sts.sequence_length, value_y_idx])
            else:
                nX.append(x[i:i+self.sts.sequence_length])
                nY.append(y[i+1:i+1+self.sts.sequence_length])
        nx = np.array(nX)
        ny = np.array(nY)
        return nx, ny
    
    def trajectory_generator(self, x, y):
        """
        Generates a trajectory of data to be fed to the network in a Seq2Seq fashion.
        Input:
           x: The inputs in the form [N x Input_dim] or [N x 1 + Input_dim] 
           y: The outputs in the form [N x Output_dim] or [N x 1 + Output_dim]
        Output:
           nx: Generated trajectory in the form [N x Trajectory_size + Sequence_size x Input_dim]
           ny: Generated trajectory in the form [N x Trajectory_size + Sequence_size x Output_dim]
        """
        nX = []
        nY = []
        
        # Stores the indices of all the variables but the continuity index
        value_x_idx = [xx for xx in range(x.shape[1])if xx!=self.sts.continuity_idx]
        value_y_idx = [xx for xx in range(y.shape[1])if xx!=self.sts.continuity_idx]
        
        # x is a sequence, y is the data-point or a sequence right after the sequence used as input
        for i in range(x.shape[0]-1-self.sts.sequence_length-self.sts.trajectory_length):
            # First check continuity of the sequence if the flag is enabled
            if not (self.sts.continuity_idx is None):
                # 1 sequence is continuous 0 otherwise.
                vx = x[i:i+self.sts.sequence_length+self.sts.trajectory_length, self.sts.continuity_idx]
                vy = y[i+1:i+1+self.sts.sequence_length+self.sts.trajectory_length, self.sts.continuity_idx]
                # Check sequence is fine, if not skip sequence.
                if ((np.max(vx) > 1) or (np.max(vy) > 1)):
                    continue
                else:
                    nX.append(x[i:i+self.sts.sequence_length+self.sts.trajectory_length, value_x_idx])
                    nY.append(y[i+1:i+1+self.sts.trajectory_length+self.sts.sequence_length, value_y_idx])
            else:
                nX.append(x[i:i+self.sts.sequence_length+self.sts.trajectory_length])
                nY.append(y[i+self.sts.sequence_length+1:i+1+self.sts.sequence_length+self.sts.trajectory_length])
        nx = np.array(nX)
        ny = np.array(nY)
        return nx, ny
