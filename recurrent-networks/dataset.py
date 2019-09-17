import h5py as h5
import numpy as np
import os

from settings import Settings

class DataPreProcessor:
    '''
    Loads the requested data, normalizes it and updates some of the settings.
    '''
    def __init__(self, settings):
        ### RAW DATA ###
        self.data = None
        self.settings = settings
        ### PROCESSING DATA ###
        self.std = None
        self.mean = None
        ### FINAL DATA ###
        self.train = None
        self.test = None
        self.val = None
        
        self.roll_count = 0

        ### RUN ###
        self.make_dataset()

    def get_data(self, path):
        # Not ideal should ask for a memory-reset at the junction of the files.
        # Will do yet it degrades training performance.
        path_list = [os.path.join(path,f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        path_list.sort()
        i = 0
        for path in path_list:
            data_chunk = self.boat_data_reader(path)
            if i == 0:
                data = data_chunk
                i += 1 
            else:
                data = np.concatenate((data,data_chunk),axis=0)
        return data

    def boat_data_reader(self, path):
        raw = h5.File(path,'r')
        data = raw['train_X']
        if data.shape[1] == 8:
            return data
        data = data[:,1:]
        return data
    
    def split_train_val_test(self, data_train, data_test, data_val):
        train_size = data_train.shape[0]
        max_batch_train = train_size//self.settings.batch_size//self.settings.sequence_length
        train_size = max_batch_train*self.settings.batch_size*self.settings.sequence_length
        self.train = data_train[:train_size]
        test_size = data_test.shape[0]
        max_batch_test = test_size//self.settings.batch_size//self.settings.sequence_length
        test_size = max_batch_test*self.settings.batch_size*self.settings.sequence_length
        self.test = data_test[:test_size]
        val_size = data_val.shape[0]
        max_batch_val = val_size//self.settings.batch_size//self.settings.sequence_length
        val_size = max_batch_val*self.settings.batch_size*self.settings.sequence_length
        self.val = data_val[:val_size]

    def extract_normalization_param(self):
        self.mean = np.mean(self.train, axis = 0)
        self.std = np.std(self.train, axis = 0)
        np.save(os.path.join(self.settings.output_dir,'means.npy'),self.mean)
        np.save(os.path.join(self.settings.output_dir,'std.npy'),self.std)

    def normalize(self):
        self.train = (self.train - self.mean)/self.std
        self.test = (self.test - self.mean)/self.std
        self.val = (self.val - self.mean)/self.std
    
    def make_dataset(self):
        path = self.settings.train_dir
        data_train = self.get_data(path)
        path = self.settings.test_dir
        data_test = self.get_data(path)
        path = self.settings.val_dir
        data_val = self.get_data(path)

        print('Splitting data ...')
        self.split_train_val_test(data_train, data_test, data_val)
        print('Checking sizes...')
        print(' + Test samples       : '+str(len(self.test)))
        print(' + Train samples      : '+str(len(self.train)))
        print(' + Validation samples : '+str(len(self.val)))
        print('Normalizing...')
        self.extract_normalization_param()
        self.normalize()

    def generate_fixed_data(self):
        raise Exception('Not implemented')

    def generate_dynamic_data(self):
        x_train = self.train
        self.rolled_train = self.train.copy()
        x_test = self.test
        x_val = self.val

        y_train = x_train[:, 0:self.settings.output_dim]
        y_test = x_test[:, 0:self.settings.output_dim]
        y_val = x_val[:, 0:self.settings.output_dim]
        y_train = np.roll(y_train, -self.settings.output_dim)
        y_test = np.roll(y_test, -self.settings.output_dim)
        y_val = np.roll(y_val, -self.settings.output_dim)
 
        x_train = x_train[:-self.settings.batch_size,:]
        y_train = y_train[:-self.settings.batch_size,:]
        # Just to give a rough idea can't really compare with CNN and MLP
        # since here we process by sequence
        x_test = x_test[:-self.settings.batch_size,:]
        y_test = y_test[:-self.settings.batch_size,:]
        x_val = x_val[:-self.settings.batch_size,:]
        y_val = y_val[:-self.settings.batch_size,:]
        
        x_train = x_train.reshape((self.settings.batch_size, -1, self.settings.input_dim))
        x_test = x_test.reshape((self.settings.batch_size, -1, self.settings.input_dim))
        x_val = x_val.reshape((self.settings.batch_size, -1, self.settings.input_dim))
        y_train = y_train.reshape((self.settings.batch_size, -1, self.settings.output_dim))
        y_test = y_test.reshape((self.settings.batch_size, -1, self.settings.output_dim))
        y_val = y_val.reshape((self.settings.batch_size, -1, self.settings.output_dim))
        return (x_train, y_train, x_test, y_test, x_val, y_val)

    def shift_train(self):
        self.x_train = self.rolled_train
        if self.roll_count == self.settings.sequence_length - 1:
            self.roll_count = 0
            self.rolled_train = self.train.copy()
            #print("Reseting shift")
        #Roll-data
        #print("Shifting data")
        # Ugly should remove the data that flows back but can't do it reshape
        # fails, need to move reformating code down here...
        self.x_train = np.roll(self.x_train, self.settings.input_dim)#[self.settings.input_dim:]
        self.roll_count += 1
        y_train = self.x_train[:, 0:self.settings.output_dim]
        y_train = np.roll(y_train, -self.settings.output_dim)
 
        x_train = self.x_train.copy()[:-self.settings.batch_size,:]
        y_train = y_train[:-self.settings.batch_size,:]
        
        x_train = x_train.reshape((self.settings.batch_size, -1, self.settings.input_dim))
        y_train = y_train.reshape((self.settings.batch_size, -1, self.settings.output_dim))
        
        return x_train, y_train
