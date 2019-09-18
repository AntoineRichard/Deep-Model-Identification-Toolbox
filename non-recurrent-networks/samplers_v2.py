import numpy as np

class UniformSampler:
    """
    A uniform sampler based on shuffling and epochs.
    """
    def __init__(self, Dataset):
        self.DS = Dataset
        
    def sample(self, x, y, batch_size):
        """
        Sample function: Create a sampler object that lasts
        for an epoch. After that it needs to be refreshed (python2)
        or regerated (python3).
        Usage:
          Create object: sampler = self.sample(x,y,bs)
          Request next batch: iteration, bx, by = next(sampler)
          Refresh object: Re-create the object.
          A try catch loop makes this process more convenient.

        Input:
            x:  the full input dataset 
            y:  the full target dataset
            bs: the desired size of the outputed batches
        Output:
            A generator object (see usage)
        """
        # Shuffle
        s = np.arange(x.shape[0])
        np.random.shuffle(s)
        x = x[s].copy()
        y = y[s].copy()
        # Generates batches
        X = []
        Y = []
        # If batch_size is None then return the whole dataset

        for i in range(int(x.shape[0]/batch_size)):
            X.append(x[i*batch_size:(i+1)*batch_size,:,:])
            Y.append(y[i*batch_size:(i+1)*batch_size,:])
        x = np.array(X)
        y = np.array(Y)
        max_iter = x.shape[0]
        # Yield based loop: Generator
        for i in range(max_iter):
            yield [i*1./max_iter, x[i],y[i]]

    def shuffle(self, x, y):
        s = np.arange(x.shape[0])
        np.random.shuffle(s)
        x = x[s].copy()
        y = y[s].copy()
        return 1.0, x, y


    def sample_train_batch(self, bs):
        """
        This function returns the train batch along with the percentage completion
        of the epoch: epoch_completion, Batch_x, Batch_y.
        """
        try:
            return next(self.TB)
        except:
            self.TB = self.sample(self.DS.train_x, self.DS.train_y, bs)
            return next(self.TB)
    
    def sample_eval_train_batch(self, bs):
        """
        This function returns the train batch along with the percentage completion
        of the epoch: epoch_completion, Batch_x, Batch_y.
        """
        if bs is None:
            return self.shuffle(self.DS.val_x, self.DS.val_y)
        try:
            return next(self.TBE)
        except:
            self.TBE = self.sample(self.DS.train_x, self.DS.train_y, bs)
            return next(self.TBE)
     
    def sample_test_batch(self, bs=None):
        """
        This function returns the test batch along with the percentage completion
        of the epoch: epoch_completion, Batch_x, Batch_y.
        """
        if bs is None:
            return self.shuffle(self.DS.val_x, self.DS.val_y)
        try:
            return next(self.TeB)
        except:
            self.TeB = self.sample(self.DS.test_x, self.DS.test_y, bs)
            return next(self.TeB)

    def sample_val_batch(self, bs=None):
        """
        This function returns the val batch along with the percentage completion
        of the epoch: epoch_completion, Batch_x, Batch_y.
        """
        if bs is None:
            return self.shuffle(self.DS.val_x, self.DS.val_y)
        try:
            return next(self.TvB)
        except:
            self.TvB = self.sample(self.DS.val_x, self.DS.val_y, bs)
            return next(self.TvB)
    
    def sample_val_trajectory(self):
        """
        This function returns the whole of the trajectory testing set.
        """
        s = np.arange(self.DS.val_traj_size)
        np.random.shuffle(s)
        x = self.DS.val_traj_x[s]
        y = self.DS.val_traj_y[s]
        return x, y
    
    def sample_test_trajectory(self):
        """
        This function returns the whole of the trajectory validation set.
        """
        s = np.arange(self.DS.test_traj_size)
        np.random.shuffle(s)
        x = self.DS.test_traj_x[s]
        y = self.DS.test_traj_y[s]
        return x, y

class PERSampler(UniformSampler):
    """
    A sampler based on the gradient prioritized experience replay
    priorization scheme.
    """
    def __init__(self, Dataset, alpha = 0.6, beta = 0.4, e = 0.0000001):
        super(PERSampler, self).__init__()
        self.DS = Dataset
        
        # PER related parameters
        self.sample_weigth = np.ones(self.DS.test_size)
        self.P = np.ones(self.DS.test_size)
        self.e = e
        self.alpha = alpha
        self.beta = beta

    def update_weights(self, loss):
        """
        This function updates the priorization weights and sampling
        probabilities.
        """
        Err = np.sqrt(loss) + self.e
        V = np.power(Err,self.alpha)
        self.P = V/np.sum(V)
        self.sample_weigth = np.power(len(self.P)*self.P,-self.beta)

    def sample_for_update(self):
        x = self.DS.train_x
        y = self.DS.train_y
        return x, y

    def sample_train_batch(self, batch_size):
        while True:
            idxs = np.random.choice(self.DS.train_size, batch_size, p=self.P)
            x = DS.train_x[idxs]
            y = DS.train_y[idxs]
            yield [x, y]

class GRADSampler(UniformSampler):
    """
    A sampler based on the gradient upper-bound sample priorization scheme.
    """
    def __init__(self, Dataset):
        super(PERSampler, self).__init__()
        self.DS = Dataset

    def sample_train_batch(self, idxs):
        x = self.DS.train_x[idxs]
        y = self.DS.train_y[idxs]
        return x, y

    def sample_superbatch(self, superbatch_size):
        try:
            return next(self.SS)
        except:
            self.SS = sample(superbatch_size)
            return next(self.SS)

    def score_batch(self, superbatch_size, batch_size, score):
        p = score/np.sum(score)
        ids = np.random.choice(superbatch_size, batch_size, p=p[0])
        idxs = self.original_indices[ids]
        probs = p[0][ids]*superbatch_size
        return idxs, 1./probs

