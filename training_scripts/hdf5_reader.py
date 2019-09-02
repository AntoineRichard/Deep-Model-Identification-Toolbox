import os
import h5py as h5
import numpy as np

# from https://stackoverflow.com/questions/15722324/sliding-window-in-numpy
def window_stack(a, width, stepsize=1):
    return np.hstack( a[i:1+i-width or None:stepsize] for i in range(0,width) )



def window_stack(a, width, stepsize=1):
    return np.hstack( a[i:1+i-width or None:stepsize] for i in range(0,width) )

def extract(data,n,p):
    new_data = window_stack(data, width = n + p, stepsize = 1)
    return new_data

class SimpleReader:

    def __init__(self,filename):
        hf = h5.File(filename,'r')
        dataset = hf["train_X"]
        self.dataset =  dataset
        self.sampleSize = dataset.shape[1]
        self.unitSize = 9

    def extract(self):
        vlin_index = range(3,self.sampleSize,self.unitSize)
        vrot_index = range(4,self.sampleSize,self.unitSize)
        cmdlin_index = range(6,self.sampleSize,self.unitSize)
        cmdrot_index = range(7,self.sampleSize,self.unitSize)

        index_all = list(vlin_index) + list(vrot_index) + list(cmdlin_index) + list(cmdrot_index)
        index_all.sort()

        return self.dataset[:,index_all]

    def extract_ARMA(self, value, p, q, freq):
        """
            Builds the A, b matrix for ARMA as well as the time + the 2 command vectors.
            p = number of data samples coming from 'value'
            q = number of data samples coming from the 2 commands
            value = index de la composante que l'on modelise : 3 pour vlin 4 pour vrot
            freq = undersampling rate
        """


        b = self.dataset[:,value]
        t = self.dataset[:,0]
        #time conversion
        t = t*10**-6#from nano to milli second
        t = t - t[0]#begining at t = 0

        A = np.zeros((len(self.dataset), p+2*q), float)
        for j in range(p):
            A[:,j] = self.dataset[:,value + (j+1)*self.unitSize*freq]
        for j in range(q):
            A[:,j+p] = self.dataset[:,15 + j*self.unitSize*freq]#15 = indexcmd_lin (6) + unitSize(9)
            A[:,j+p+q] = self.dataset[:,16 + j*self.unitSize*freq]#16 = indexcmd_lin (7) + unitSize(9)

        cmd_linear_x = self.dataset[:,6]
        cmd_angular_z = self.dataset[:,7]

        return A, b, t, cmd_linear_x, cmd_angular_z

class ReaderBoatKingfisher:
    """
    handling reading of the drone dataset in hdf5
    splitting data into test/val/train
    """

    def __init__(self, filenames, input_history, input_dim, output_forecast, output_dim, cmd_dim, split_test_index, split_val_index):

        #testing wether filenames is a list or str

        self.input_state_dim = input_dim
        self.output_state_dim = output_dim
        self.cmd_dim = cmd_dim
        self.input_history = input_history
        self.output_forecast = output_forecast

        self.trainingData_xy, self.validatingData_xy, self.testingData_xy = self.setupData(filenames, split_test_index, split_val_index)
        
        #self.normalize()
        self.train_size = len(self.trainingData_xy)
        self.test_size = len(self.testingData_xy)
        self.val_size = len(self.validatingData_xy)
        import random
        L = list(range(0,self.train_size))
        random.shuffle(L)
        self.trainingData_xy = self.trainingData_xy[L]

    def normalization(self, x,std=None,mean=None):
        if not std:
            tmp = []
            std = np.std(x,axis=0)#standard deviation
            for i in range(0,5):
              tmp.append(np.mean(std[i::5]))
            std = np.array(tmp)
    
        if not mean:
            tmp = []
            mean = np.mean(x,axis=0)
            for i in range(0,5):
              tmp.append(np.mean(mean[i::5]))
            mean = np.array(tmp)
        X_scaled = x
        for i in range(0,len(std)):
            if std[i]>0.0000001:#non constant feature
                X_scaled[:,i::5] = (x[:,i::5] - mean[i])/std[i]
            else:
                X_scaled[:,i::5] = x[:,i::5] - mean[i]
    
            x = X_scaled
    
        return x, std, mean

    def remove_timestamp(self, a):
        #return np.array([np.delete(x,np.s_[0::6]) for x in a[:]])
        return np.delete(a,np.s_[0::6],1)

    def setupData(self, directories, test_index, val_index):
        #split filenames into meta_train_files, test_file
        if os.path.isdir(directories[0]):
            root = directories[0]
            filenames = os.listdir(directories[0])
        for i, filename in enumerate(filenames):
            if i == 0:
                data_xy = self.loadData(os.path.join(root,filename))
                data_xy = extract(data_xy, 99, 1)
            else:
                tmp_xy = self.loadData(os.path.join(root,filename))
                tmp_xy = extract(tmp_xy, 99, 1)
                data_xy = np.concatenate((data_xy, tmp_xy),axis = 0)
        
        if os.path.isdir(directories[1]):
            root = directories[1]
            filenames = os.listdir(directories[1])
        for i, filename in enumerate(filenames):
            if i == 0:
                data_test_xy = self.loadData(os.path.join(root,filename))
                data_test_xy = extract(data_test_xy, 99, 1)
            else:
                tmp_xy = self.loadData(os.path.join(root,filename))
                tmp_xy = extract(tmp_xy, 99, 1)
                data_test_xy = np.concatenate((data_test_xy, tmp_xy),axis = 0)
        
        if os.path.isdir(directories[2]):
            root = directories[2]
            filenames = os.listdir(directories[2])
        for i, filename in enumerate(filenames):
            if i == 0:
                data_val_xy = self.loadData(os.path.join(root,filename))
                data_val_xy = extract(data_val_xy, 99, 1)
            else:
                tmp_xy = self.loadData(os.path.join(root,filename))
                tmp_xy = extract(tmp_xy, 99, 1)
                data_val_xy = np.concatenate((data_val_xy, tmp_xy),axis = 0)

        data_size = data_xy.shape[0] + data_test_xy.shape[0] + data_val_xy[0]
        test_size = data_test_xy.shape[0]
        train_size = data_xy.shape[0]
        val_size = data_val_xy.shape[0]
        print('Removing Timestamps ...')
        print('Splitting data ...')
        train_xy = data_xy 
        test_xy = data_test_xy
        val_xy = data_val_xy
        print('Checking sizes...')
        print(' + Test samples       : '+str(len(test_xy)))
        print(' + Train samples      : '+str(len(train_xy)))
        print(' + Validation samples : '+str(len(val_xy)))
        print('Normalizing ...')
        data_xy = self.normalize(test_xy) # Single normalization
        data_xy = self.normalize(val_xy) # Single normalization
        data_xy = self.normalize(train_xy) # Single normalization
        print('Done !')

        return train_xy, val_xy, test_xy

    def loadData(self,name):
        """
        loadfile of chronological & continuous state succession store in "train_x" dataset
        """

        hf = h5.File(name,'r')
        dset_X = hf["train_X"][:,:]
        dset_size = dset_X.shape[0]
        training_xy = self.remove_timestamp(dset_X)
        hf.close()

        return training_xy


    def normalize(self,data):
        data,  self.std, self.mean = self.normalization(data)


    def getTrain(self,li):
        tmp_x = self.trainingData_xy[li,:self.input_history * self.input_state_dim]
        tmp_y = self.trainingData_xy[li,self.input_history * self.input_state_dim:(self.input_history+self.output_forecast)* self.input_state_dim]
        tmp_y = np.delete(tmp_y,np.s_[3::5],1)
        tmp_y = np.delete(tmp_y,np.s_[3::4],1)
        return tmp_x, tmp_y

    def getTrainRNN(self,li):
        tmp_x = self.trainingData_xy[li,:self.input_history * self.input_state_dim]
        tmp_y = self.trainingData_xy[li,self.input_history * self.input_state_dim:(self.input_history+self.output_forecast)* self.input_state_dim]
        return tmp_x, tmp_y

    def getTest(self,li):
        tmp_x = self.testingData_xy[li,:self.input_history * self.input_state_dim]
        tmp_y = self.testingData_xy[li,self.input_history * self.input_state_dim:(self.input_history+self.output_forecast)* self.input_state_dim]
        tmp_y = np.delete(tmp_y,np.s_[3::5],1)
        tmp_y = np.delete(tmp_y,np.s_[3::4],1)
        return tmp_x, tmp_y

    def getTestRNN(self,li):
        tmp_x = self.testingData_xy[li,:self.input_history * self.input_state_dim]
        tmp_y = self.testingData_xy[li,self.input_history * self.input_state_dim:(self.input_history+self.output_forecast)* self.input_state_dim]
        return tmp_x, tmp_y

    def getVal(self,li):
        tmp_x = self.validatingData_xy[li,:self.input_history * self.input_state_dim]
        tmp_y = self.validatingData_xy[li,self.input_history * self.input_state_dim:(self.input_history+self.output_forecast)* self.input_state_dim]
        tmp_y = np.delete(tmp_y,np.s_[3::5],1)
        tmp_y = np.delete(tmp_y,np.s_[3::4],1)
        return tmp_x, tmp_y

    def getValRNN(self,li):
        tmp_x = self.validatingData_xy[li,:self.input_history * self.input_state_dim]
        tmp_y = self.validatingData_xy[li,self.input_history * self.input_state_dim:(self.input_history+self.output_forecast)* self.input_state_dim]
        tmp_y = np.delete(tmp_y,np.s_[3::5],1)
        tmp_y = np.delete(tmp_y,np.s_[3::4],1)
        return tmp_x, tmp_y

    def getWindowTest(self,size):
        idx = int(np.random.rand(1)*self.test_size)
        tmp_x = self.testingData_xy[idx,:(size+self.input_history) * self.input_state_dim]
        tmp_y = self.testingData_xy[idx,(self.input_history + size) * self.input_state_dim:(self.input_history + self.output_forecast)* self.input_state_dim]
        tmp_y = np.delete(tmp_y,np.s_[3::5],1)
        tmp_y = np.delete(tmp_y,np.s_[3::4],1)
        return tmp_x, tmp_y

    def getWindowTestRNN(self,size):
        idx = int(np.random.rand(1)*self.test_size)
        tmp_x = self.testingData_xy[idx,:(size+self.input_history) * self.input_state_dim]
        tmp_y = self.testingData_xy[idx,(self.input_history + size) * self.input_state_dim:(self.input_history + self.output_forecast)* self.input_state_dim]
        return tmp_x, tmp_y

    def getWindowVal(self,size):
        idx = int(np.random.rand(1)*self.val_size)
        tmp_x = self.validatingData_xy[idx,:]
        tmp_x = np.reshape(tmp_x, [-1,self.input_state_dim])
        tmp_x = window_stack(tmp_x, self.input_history + self.output_forecast)
        tmp_y = tmp_x[1:,:]
        return tmp_x[:size,:], tmp_y[:size,:]

    def getWindowValRNN(self,size):
        idx = int(np.random.rand(1)*self.val_size)
        tmp_x = self.validatingData_xy[idx,:(size+self.input_history) * self.input_state_dim]
        tmp_y = self.validatingData_xy[idx,(self.input_history + size) * self.input_state_dim:(self.input_history + self.output_forecast)* self.input_state_dim]
        tmp_y = np.delete(tmp_y,np.s_[3::5],1)
        tmp_y = np.delete(tmp_y,np.s_[3::4],1)
        return tmp_x, tmp_y


class ReaderDroneETHZ:
    """
    handling reading of the drone dataset in hdf5
    splitting data into test/val/train
    """

    def __init__(self, filenames, input_history, input_dim, output_forecast, output_dim, cmd_dim, split_test_index, split_val_index):

        #testing wether filenames is a list or str

        self.input_state_dim = input_dim
        self.output_state_dim = output_dim
        self.cmd_dim = cmd_dim
        self.input_history = input_history
        self.output_forecast = output_forecast

        self.trainingData_xy, self.validatingData_xy, self.testingData_xy = self.setupData(filenames, split_test_index, split_val_index)
        
        #self.normalize()
        self.train_size = len(self.trainingData_xy)
        self.test_size = len(self.testingData_xy)
        self.val_size = len(self.validatingData_xy)
        import random
        L = list(range(0,self.train_size))
        random.shuffle(L)
        self.trainingData_xy = self.trainingData_xy[L]
    
    def normalization(self, x,std=None,mean=None):
        if not std:
            tmp = []
            std = np.std(x,axis=0)#standard deviation
            for i in range(0,13):
              tmp.append(np.mean(std[i::13]))
            std = np.array(tmp)
    
        if not mean:
            tmp = []
            mean = np.mean(x,axis=0)
            for i in range(0,13):
              tmp.append(np.mean(mean[i::13]))
            mean = np.array(tmp)
        X_scaled = x
        for i in range(0,len(std)):
            if std[i]>0.0000001:#non constant feature
                X_scaled[:,i::13] = (x[:,i::13] - mean[i])/std[i]
            else:
                X_scaled[:,i::13] = x[:,i::13] - mean[i]
    
            x = X_scaled
    
        return x, std, mean

    def remove_timestamp(self, a):
        #return np.array([np.delete(x,np.s_[0::6]) for x in a[:]])
        return np.delete(a,np.s_[0::14],1)

    def setupData(self, filenames, test_index, val_index):
        #split filenames into meta_train_files, test_file
        root = filenames[0]
        filenames = os.listdir(root)
        for i, filename in enumerate(filenames):
            if i == 0:
                data_xy = self.loadData(os.path.join(root,filename))
                data_xy = extract(data_xy, 99, 1)
            else:
                tmp_xy = self.loadData(os.path.join(root,filename))
                tmp_xy = extract(tmp_xy, 99, 1)
                data_xy = np.concatenate((data_xy, tmp_xy),axis = 0)
        
        data_size = data_xy.shape[0] 
        test_size = int(data_size*1/10)
        train_size = int(data_size*8/10)
        val_size = int(data_size*1/10)
        print('Evaluating sizes...')
        print(' + Test samples       : '+str(test_size))
        print(' + Train samples      : '+str(train_size))
        print(' + Validation samples : '+str(val_size))
        print('Removing Timestamps ...')
        print('Splitting data ...')
        train_xy = data_xy[:train_size,:] 
        test_xy = data_xy[train_size:train_size+test_size,:]
        val_xy = data_xy[train_size+test_size:train_size+test_size+val_size,:]
        print('Checking sizes...')
        print(' + Test samples       : '+str(len(test_xy)))
        print(' + Train samples      : '+str(len(train_xy)))
        print(' + Validation samples : '+str(len(val_xy)))
        print('Normalizing ...')
        print(train_xy.shape)
        data_xy = self.normalize(test_xy) # Single normalization
        data_xy = self.normalize(val_xy) # Single normalization
        data_xy = self.normalize(train_xy) # Single normalization
        print('Done !')

        return train_xy, val_xy, test_xy

    def loadData(self,name):
        """
        loadfile of chronological & continuous state succession store in "train_x" dataset
        """

        dset_X = np.load(name)
        dset_X = np.swapaxes(dset_X,0,1)
        training_xy = self.remove_timestamp(dset_X)

        return training_xy


    def normalize(self,data):
        data,  self.std, self.mean = self.normalization(data)


    def getTrain(self,li):
        tmp_x = self.trainingData_xy[li,:self.input_history * self.input_state_dim]
        tmp_y = self.trainingData_xy[li,self.input_history * self.input_state_dim:(self.input_history+self.output_forecast)* self.input_state_dim]
        tmp_y = np.delete(tmp_y,np.s_[12::13],1)
        tmp_y = np.delete(tmp_y,np.s_[11::12],1)
        tmp_y = np.delete(tmp_y,np.s_[10::11],1)
        tmp_y = np.delete(tmp_y,np.s_[9::10],1)
        tmp_y = np.delete(tmp_y,np.s_[8::9],1)
        tmp_y = np.delete(tmp_y,np.s_[7::8],1)
        return tmp_x, tmp_y

    def getTrainRNN(self,li):
        tmp_x = self.trainingData_xy[li,:self.input_history * self.input_state_dim]
        tmp_y = self.trainingData_xy[li,self.input_history * self.input_state_dim:(self.input_history+self.output_forecast)* self.input_state_dim]
        return tmp_x, tmp_y

    def getTest(self,li):
        tmp_x = self.testingData_xy[li,:self.input_history * self.input_state_dim]
        tmp_y = self.testingData_xy[li,self.input_history * self.input_state_dim:(self.input_history+self.output_forecast)* self.input_state_dim]
        tmp_y = np.delete(tmp_y,np.s_[12::13],1)
        tmp_y = np.delete(tmp_y,np.s_[11::12],1)
        tmp_y = np.delete(tmp_y,np.s_[10::11],1)
        tmp_y = np.delete(tmp_y,np.s_[9::10],1)
        tmp_y = np.delete(tmp_y,np.s_[8::9],1)
        tmp_y = np.delete(tmp_y,np.s_[7::8],1)
        return tmp_x, tmp_y

    def getTestRNN(self,li):
        tmp_x = self.testingData_xy[li,:self.input_history * self.input_state_dim]
        tmp_y = self.testingData_xy[li,self.input_history * self.input_state_dim:(self.input_history+self.output_forecast)* self.input_state_dim]
        return tmp_x, tmp_y

    def getVal(self,li):
        tmp_x = self.validatingData_xy[li,:self.input_history * self.input_state_dim]
        tmp_y = self.validatingData_xy[li,self.input_history * self.input_state_dim:(self.input_history+self.output_forecast)* self.input_state_dim]
        tmp_y = np.delete(tmp_y,np.s_[12::13],1)
        tmp_y = np.delete(tmp_y,np.s_[11::12],1)
        tmp_y = np.delete(tmp_y,np.s_[10::11],1)
        tmp_y = np.delete(tmp_y,np.s_[9::10],1)
        tmp_y = np.delete(tmp_y,np.s_[8::9],1)
        tmp_y = np.delete(tmp_y,np.s_[7::8],1)
        return tmp_x, tmp_y

    def getValRNN(self,li):
        tmp_x = self.validatingData_xy[li,:self.input_history * self.input_state_dim]
        tmp_y = self.validatingData_xy[li,self.input_history * self.input_state_dim:(self.input_history+self.output_forecast)* self.input_state_dim]
        tmp_y = np.delete(tmp_y,np.s_[12::13],1)
        tmp_y = np.delete(tmp_y,np.s_[11::12],1)
        tmp_y = np.delete(tmp_y,np.s_[10::11],1)
        tmp_y = np.delete(tmp_y,np.s_[9::10],1)
        tmp_y = np.delete(tmp_y,np.s_[8::9],1)
        tmp_y = np.delete(tmp_y,np.s_[7::8],1)
        return tmp_x, tmp_y

    def getWindowTest(self,size):
        idx = int(np.random.rand(1)*self.test_size)
        tmp_x = self.testingData_xy[idx,:(size+self.input_history) * self.input_state_dim]
        tmp_y = self.testingData_xy[idx,(self.input_history + size) * self.input_state_dim:(self.input_history + self.output_forecast)* self.input_state_dim]
        tmp_y = np.delete(tmp_y,np.s_[12::13],1)
        tmp_y = np.delete(tmp_y,np.s_[11::12],1)
        tmp_y = np.delete(tmp_y,np.s_[10::11],1)
        tmp_y = np.delete(tmp_y,np.s_[9::10],1)
        tmp_y = np.delete(tmp_y,np.s_[8::9],1)
        tmp_y = np.delete(tmp_y,np.s_[7::8],1)
        return tmp_x, tmp_y

    def getWindowTestRNN(self,size):
        idx = int(np.random.rand(1)*self.test_size)
        tmp_x = self.testingData_xy[idx,:(size+self.input_history) * self.input_state_dim]
        tmp_y = self.testingData_xy[idx,(self.input_history + size) * self.input_state_dim:(self.input_history + self.output_forecast)* self.input_state_dim]
        return tmp_x, tmp_y

    def getWindowVal(self,size):
        idx = int(np.random.rand(1)*self.val_size)
        tmp_x = self.validatingData_xy[idx,:]
        tmp_x = np.reshape(tmp_x, [-1,self.input_state_dim])
        tmp_x = window_stack(tmp_x, self.input_history + self.output_forecast)
        tmp_y = tmp_x[1:,:]
        return tmp_x[:size,:], tmp_y[:size,:]

    def getWindowValRNN(self,size):
        idx = int(np.random.rand(1)*self.val_size)
        tmp_x = self.validatingData_xy[idx,:(size+self.input_history) * self.input_state_dim]
        tmp_y = self.validatingData_xy[idx,(self.input_history + size) * self.input_state_dim:(self.input_history + self.output_forecast)* self.input_state_dim]
        tmp_y = np.delete(tmp_y,np.s_[12::13],1)
        tmp_y = np.delete(tmp_y,np.s_[11::12],1)
        tmp_y = np.delete(tmp_y,np.s_[10::11],1)
        tmp_y = np.delete(tmp_y,np.s_[9::10],1)
        tmp_y = np.delete(tmp_y,np.s_[8::9],1)
        tmp_y = np.delete(tmp_y,np.s_[7::8],1)
        return tmp_x, tmp_y

class ReaderDrone:
    """
    handling reading of the drone dataset in hdf5
    splitting data into test/val/train
    """

    def __init__(self, filenames, input_history, input_dim, output_forecast, output_dim, cmd_dim, split_test_index, split_val_index):

        #testing wether filenames is a list or str

        self.input_state_dim = input_dim
        self.output_state_dim = output_dim
        self.cmd_dim = cmd_dim
        self.input_history = input_history
        self.output_forecast = output_forecast

        self.trainingData_xy, self.validatingData_xy, self.testingData_xy = self.setupData(filenames, split_test_index, split_val_index)
        
        #self.normalize()
        self.train_size = len(self.trainingData_xy)
        self.test_size = len(self.testingData_xy)
        self.val_size = len(self.validatingData_xy)
        import random
        L = list(range(0,self.train_size))
        random.shuffle(L)
        self.trainingData_xy = self.trainingData_xy[L]

    def normalization(self, x,std=None,mean=None):
        if not std:
            tmp = []
            std = np.std(x,axis=0)#standard deviation
            for i in range(0,8):
              tmp.append(np.mean(std[i::8]))
            std = np.array(tmp)
    
        if not mean:
            tmp = []
            mean = np.mean(x,axis=0)
            for i in range(0,8):
              tmp.append(np.mean(mean[i::8]))
            mean = np.array(tmp)
        X_scaled = x
        for i in range(0,len(std)):
            if std[i]>0.0000001:#non constant feature
                X_scaled[:,i::8] = (x[:,i::8] - mean[i])/std[i]
            else:
                X_scaled[:,i::8] = x[:,i::8] - mean[i]
    
            x = X_scaled
    
        return x, std, mean

    def remove_timestamp(self, a):
        #return np.array([np.delete(x,np.s_[0::6]) for x in a[:]])
        return a#np.delete(a,np.s_[0::6],1)

    def setupData(self, directories, test_index, val_index):
        #split filenames into meta_train_files, test_file
        if os.path.isdir(directories[0]):
            root = directories[0]
            filenames = os.listdir(directories[0])
            for i, filename in enumerate(filenames):
                if i == 0:
                    data_xy = self.loadData(os.path.join(root,filename))
                    data_xy = extract(data_xy, 99, 1)
                else:
                    tmp_xy = self.loadData(os.path.join(root,filename))
                    tmp_xy = extract(tmp_xy, 99, 1)
                    data_xy = np.concatenate((data_xy, tmp_xy),axis = 0)
        
        if os.path.isdir(directories[1]):
            root = directories[1]
            filenames = os.listdir(directories[1])
            for i, filename in enumerate(filenames):
                if i == 0:
                    data_test_xy = self.loadData(os.path.join(root,filename))
                    data_test_xy = extract(data_test_xy, 99, 1)
                else:
                    tmp_xy = self.loadData(os.path.join(root,filename))
                    tmp_xy = extract(tmp_xy, 99, 1)
                    data_test_xy = np.concatenate((data_test_xy, tmp_xy),axis = 0)
        
        if os.path.isdir(directories[2]):
            root = directories[2]
            filenames = os.listdir(directories[2])
            for i, filename in enumerate(filenames):
                if i == 0:
                    data_val_xy = self.loadData(os.path.join(root,filename))
                    data_val_xy = extract(data_val_xy, 99, 1)
                else:
                    tmp_xy = self.loadData(os.path.join(root,filename))
                    tmp_xy = extract(tmp_xy, 99, 1)
                    data_val_xy = np.concatenate((data_val_xy, tmp_xy),axis = 0)

        data_size = data_xy.shape[0] + data_test_xy.shape[0] + data_val_xy[0]
        test_size = data_test_xy.shape[0]
        train_size = data_xy.shape[0]
        val_size = data_val_xy.shape[0]
        print('Removing Timestamps ...')
        print('Splitting data ...')
        train_xy = data_xy 
        test_xy = data_test_xy
        val_xy = data_val_xy
        print('Checking sizes...')
        print(' + Test samples       : '+str(len(test_xy)))
        print(' + Train samples      : '+str(len(train_xy)))
        print(' + Validation samples : '+str(len(val_xy)))
        print('Normalizing ...')
        data_xy = self.normalize(test_xy) # Single normalization
        data_xy = self.normalize(val_xy) # Single normalization
        data_xy = self.normalize(train_xy) # Single normalization
        print('Done !')

        return train_xy, val_xy, test_xy

    def loadData(self,name):
        """
        loadfile of chronological & continuous state succession store in "train_x" dataset
        """

        hf = h5.File(name,'r')
        dset_X = hf["train_X"][:,:]
        dset_size = dset_X.shape[0]
        training_xy = self.remove_timestamp(dset_X)
        hf.close()

        return training_xy


    def normalize(self,data):
        data,  self.std, self.mean = self.normalization(data)


    def getTrain(self,li):
        tmp_x = self.trainingData_xy[li,:self.input_history * self.input_state_dim]
        tmp_y = self.trainingData_xy[li,self.input_history * self.input_state_dim:(self.input_history+self.output_forecast)* self.input_state_dim]
        tmp_y = np.delete(tmp_y,np.s_[7::8],1)
        tmp_y = np.delete(tmp_y,np.s_[6::7],1)
        tmp_y = np.delete(tmp_y,np.s_[5::6],1)
        tmp_y = np.delete(tmp_y,np.s_[4::5],1)
        return tmp_x, tmp_y

    def getTrainRNN(self,li):
        tmp_x = self.trainingData_xy[li,:self.input_history * self.input_state_dim]
        tmp_y = self.trainingData_xy[li,self.input_history * self.input_state_dim:(self.input_history+self.output_forecast)* self.input_state_dim]
        return tmp_x, tmp_y

    def getTest(self,li):
        tmp_x = self.testingData_xy[li,:self.input_history * self.input_state_dim]
        tmp_y = self.testingData_xy[li,self.input_history * self.input_state_dim:(self.input_history+self.output_forecast)* self.input_state_dim]
        tmp_y = np.delete(tmp_y,np.s_[7::8],1)
        tmp_y = np.delete(tmp_y,np.s_[6::7],1)
        tmp_y = np.delete(tmp_y,np.s_[5::6],1)
        tmp_y = np.delete(tmp_y,np.s_[4::5],1)
        return tmp_x, tmp_y

    def getTestRNN(self,li):
        tmp_x = self.testingData_xy[li,:self.input_history * self.input_state_dim]
        tmp_y = self.testingData_xy[li,self.input_history * self.input_state_dim:(self.input_history+self.output_forecast)* self.input_state_dim]
        return tmp_x, tmp_y

    def getVal(self,li):
        tmp_x = self.validatingData_xy[li,:self.input_history * self.input_state_dim]
        tmp_y = self.validatingData_xy[li,self.input_history * self.input_state_dim:(self.input_history+self.output_forecast)* self.input_state_dim]
        tmp_y = np.delete(tmp_y,np.s_[7::8],1)
        tmp_y = np.delete(tmp_y,np.s_[6::7],1)
        tmp_y = np.delete(tmp_y,np.s_[5::6],1)
        tmp_y = np.delete(tmp_y,np.s_[4::5],1)
        return tmp_x, tmp_y

    def getValRNN(self,li):
        tmp_x = self.validatingData_xy[li,:self.input_history * self.input_state_dim]
        tmp_y = self.validatingData_xy[li,self.input_history * self.input_state_dim:(self.input_history+self.output_forecast)* self.input_state_dim]
        tmp_y = np.delete(tmp_y,np.s_[7::8],1)
        tmp_y = np.delete(tmp_y,np.s_[6::7],1)
        tmp_y = np.delete(tmp_y,np.s_[5::6],1)
        tmp_y = np.delete(tmp_y,np.s_[4::5],1)
        return tmp_x, tmp_y

    def getWindowTest(self,size):
        idx = int(np.random.rand(1)*self.test_size)
        tmp_x = self.testingData_xy[idx,:(size+self.input_history) * self.input_state_dim]
        tmp_y = self.testingData_xy[idx,(self.input_history + size) * self.input_state_dim:(self.input_history + self.output_forecast)* self.input_state_dim]
        tmp_y = np.delete(tmp_y,np.s_[7::8],1)
        tmp_y = np.delete(tmp_y,np.s_[6::7],1)
        tmp_y = np.delete(tmp_y,np.s_[5::6],1)
        tmp_y = np.delete(tmp_y,np.s_[4::5],1)
        return tmp_x, tmp_y

    def getWindowTestRNN(self,size):
        idx = int(np.random.rand(1)*self.test_size)
        tmp_x = self.testingData_xy[idx,:(size+self.input_history) * self.input_state_dim]
        tmp_y = self.testingData_xy[idx,(self.input_history + size) * self.input_state_dim:(self.input_history + self.output_forecast)* self.input_state_dim]
        return tmp_x, tmp_y

    def getWindowVal(self,size):
        idx = int(np.random.rand(1)*self.val_size)
        tmp_x = self.validatingData_xy[idx,:]
        tmp_x = np.reshape(tmp_x, [-1,self.input_state_dim])
        tmp_x = window_stack(tmp_x, self.input_history + self.output_forecast)
        tmp_y = tmp_x[1:,:]
        return tmp_x[:size,:], tmp_y[:size,:]

    def getWindowValRNN(self,size):
        idx = int(np.random.rand(1)*self.val_size)
        tmp_x = self.validatingData_xy[idx,:(size+self.input_history) * self.input_state_dim]
        tmp_y = self.validatingData_xy[idx,(self.input_history + size) * self.input_state_dim:(self.input_history + self.output_forecast)* self.input_state_dim]
        tmp_y = np.delete(tmp_y,np.s_[7::8],1)
        tmp_y = np.delete(tmp_y,np.s_[6::7],1)
        tmp_y = np.delete(tmp_y,np.s_[5::6],1)
        tmp_y = np.delete(tmp_y,np.s_[4::5],1)
        return tmp_x, tmp_y
