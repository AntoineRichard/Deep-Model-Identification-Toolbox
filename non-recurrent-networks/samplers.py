import numpy as np



class UniformSampler:
    def __init__(self, Dataset, multistep_number):
        self.DS = Dataset
        self.valPick = None
        self.testPick = None
        self.trainPick = None
        self.reset_train()
        self.reset_test()
        self.reset_val()
        self.allowed_trajectories = None
        self.set_trajectories()
        self.msn = multistep_number
        self.msc = 0

    def set_trajectories(self):
        self.allowed_trajectories = list(range(0,2000,20))

    def reset_val(self):
        self.valPick = np.ones(self.DS.val_size)

    def reset_test(self):
        self.testPick = np.ones(self.DS.test_size)

    def reset_train(self):
        self.trainPick = np.ones(self.DS.train_size)

    def sample(self, batch_size):
        if np.sum(self.trainPick) <= batch_size:
            self.reset_train()

        p = self.trainPick/(np.sum(self.trainPick))
        idxs = np.random.choice(self.DS.train_size,
                                  batch_size,
                                  p=p,
                                  replace = False)
        self.trainPick[idxs] = 0
        return idxs

    def sample_4_test(self, batch_size):
        if np.sum(self.testPick) <= batch_size:
            self.reset_test()

        p = self.testPick/(np.sum(self.testPick))
        idxs = np.random.choice(self.DS.test_size,
                                  batch_size,
                                  p=p,
                                  replace = False)
        self.testPick[idxs] = 0
        return idxs

    def sample_4_val(self, batch_size):
        if np.sum(self.valPick) <= batch_size:
            self.reset_val()

        p = self.valPick/(np.sum(self.valPick))
        idxs = np.random.choice(self.DS.val_size,
                                  batch_size,
                                  p=p,
                                  replace = False)
        self.valPick[idxs] = 0
        return idxs


    def next_train_batch(self, batch_size):
        idxs = self.sample(batch_size)
        x, y = self.DS.getTrain(idxs)
        return x, y

    def next_train_batch_RNN(self, batch_size):
        idxs = self.sample(batch_size)
        x, y = self.DS.getTrainRNN(idxs)
        return x, y


    def next_test_batch(self, batch_size):
        idxs = self.sample_4_test(batch_size)
        x, y = self.DS.getTest(idxs)
        return x, y

    def next_test_batch_RNN(self, batch_size):
        idxs = self.sample_4_test(batch_size)
        x, y = self.DS.getTestRNN(idxs)
        return x, y

    def test_batch(self, idxs):
        x, y = self.DS.getTest(idxs)
        return x, y

    def next_val_batch(self, batch_size):
        idxs = self.sample_4_val(batch_size)
        x, y = self.DS.getVal(idxs)
        return x, y

    def next_val_batch_RNN(self, batch_size):
        idxs = self.sample_4_val(batch_size)
        x, y = self.DS.getValRNN(idxs)
        return x, y

    def val_batch(self, idxs):
        x, y = self.DS.getVal(idxs)
        return x, y

    def get_random_test_batch(self, test_window) :
        #offset = int((self.DS.test_size - test_window) * np.random.rand(1))
        if self.msc == self.msn:
            self.msc = 0
        offset = allowed_trajectories[self.msc]
        self.msc += 1
        dynamic_range = range(offset, offset + test_window)
        batch_x, batch_y = self.DS.getTest(dynamic_range)
        return batch_x, batch_y

    def get_random_test_batch_RNN(self, test_window) :
        #offset = int((self.DS.test_size - test_window) * np.random.rand(1))
        if self.msc == self.msn:
            self.msc = 0
        offset = allowed_trajectories[self.msc]
        self.msc += 1
        dynamic_range = range(offset, offset + test_window)
        batch_x, batch_y = self.DS.getTestRNN(dynamic_range)
        return batch_x, batch_y

    def get_random_val_batch(self, test_window) :
        #offset = int((self.DS.test_size - test_window) * np.random.rand(1))
        if self.msc == self.msn:
            self.msc = 0
        offset = self.allowed_trajectories[self.msc]
        self.msc += 1
        dynamic_range = range(offset, offset + test_window)
        batch_x, batch_y = self.DS.getVal(dynamic_range)
        return batch_x, batch_y

    def get_random_val_batch_RNN(self, test_window) :
        #offset = int((self.DS.val_size - test_window) * np.random.rand(1))
        if self.msc == self.msn:
            self.msc = 0
        offset = self.allowed_trajectories[self.msc]
        self.msc += 1
        dynamic_range = range(offset, offset + test_window)
        batch_x, batch_y = self.DS.getValRNN(dynamic_range)
        return batch_x, batch_y


class PrioritizingSampler:
    def __init__(self, Dataset, multistep_number, alpha = 0.6, beta = 0.4, e = 0.0000001):
        self.DS = Dataset
        self.valPick = None
        self.testPick = None
        self.trainPick = None
        self.reset_train()
        self.reset_test()
        self.reset_val()

        self.sample_weigth = np.ones(self.DS.test_size)
        self.P = np.ones(self.DS.test_size)#probability distribution over train sample
        #prioritization hyperparameter
        self.e = e
        self.alpha = alpha
        self.beta = beta
        self.allowed_trajectories = None
        self.msc = 0
        self.msn = multistep_number

    def set_trajectories(self):
        self.allowed_trajectories = list(range(0,2000,20))

    def prioritize(self,loss):
        """
        compute probability distribution over sample for prioritzed sampling
        @param : predictions : result of the model for the training dset as a whole
        """

        #compute model error

        #truth for output a 1 step, laast 2 element of the n long array the first element is further in the futur.
        Err = np.sqrt(loss) + self.e

        #Value of samples
        V = np.power(Err,self.alpha)
        #probability distribution for sampling
        self.P = V/np.sum(V)

        self.sample_weigth = np.power(len(self.P)*self.P,-self.beta)


    def reset_val(self):
        self.valPick = np.ones(self.DS.val_size)

    def reset_test(self):
        self.testPick = np.ones(self.DS.test_size)

    def reset_train(self):
        self.trainPick = np.ones(self.DS.train_size)

    def samplePrioritized(self, batch_size):

        idxs = np.random.choice(self.DS.train_size,
                                  batch_size,
                                  p=self.P)
        return idxs


    def sample(self, batch_size):
        if np.sum(self.trainPick) <= batch_size:
            print('EPOCH REACHED')
            self.reset_train()

        p = self.trainPick/(np.sum(self.trainPick))
        idxs = np.random.choice(self.DS.train_size,
                                  batch_size,
                                  p=p,
                                  replace = False)
        self.trainPick[idxs] = 0
        return idxs

    def sample_4_test(self, batch_size):
        if np.sum(self.testPick) <= batch_size:
            self.reset_valid()

        p = self.testPick/(np.sum(self.testPick))
        idxs = np.random.choice(self.DS.test_size,
                                  batch_size,
                                  p=p,
                                  replace = False)
        self.testPick[idxs] = 0
        return idxs

    def sample_4_val(self, batch_size):
        if np.sum(self.valPick) <= batch_size:
            self.reset_val()

        p = self.valPick/(np.sum(self.valPick))
        idxs = np.random.choice(self.DS.val_size,
                                  batch_size,
                                  p=p,
                                  replace = False)
        self.valPick[idxs] = 0
        return idxs


    def next_train_batch(self, batch_size):
        idxs = self.sample(batch_size)
        x, y = self.DS.getTrain(idxs)
        return x, y

    def next_train_batch_RNN(self, batch_size):
        idxs = self.sample(batch_size)
        x, y = self.DS.getTrainRNN(idxs)
        return x, y


    def next_train_batch_prioritized_multistep(self, batch_size):
        idxs = self.samplePrioritized(batch_size)
        x, y = self.DS.getTrainRNN(idxs)
        return x, y, self.sample_weigth[idxs]

    def next_train_batch_prioritized(self, batch_size):
        idxs = self.samplePrioritized(batch_size)
        x, y = self.DS.getTrain(idxs)
        return x, y, self.sample_weigth[idxs]


    def next_test_batch(self, batch_size):
        idxs = self.sample_4_test(batch_size)
        x, y = self.DS.getTest(idxs)
        return x, y

    def next_test_batch_RNN(self, batch_size):
        idxs = self.sample_4_test(batch_size)
        x, y = self.DS.getTestRNN(idxs)
        return x, y

    def test_batch(self, idxs):
        x, y = self.DS.getTest(idxs)
        return x, y

    def next_val_batch(self, batch_size):
        idxs = self.sample_4_val(batch_size)
        x, y = self.DS.getVal(idxs)
        return x, y

    def next_val_batch_RNN(self, batch_size):
        idxs = self.sample_4_val(batch_size)
        x, y = self.DS.getValRNN(idxs)
        return x, y

    def val_batch(self, idxs):
        x, y = self.DS.getVal(idxs)
        return x, y

    def get_random_test_batch(self, test_window) :
        #offset = int((self.DS.test_size - test_window) * np.random.rand(1))
        if self.msc == self.msn:
            self.msc = 0
        offset = allowed_trajectories[self.msc]
        self.msc += 1
        dynamic_range = range(offset, offset + test_window)
        batch_x, batch_y = self.DS.getTest(dynamic_range)
        return batch_x, batch_y

    def get_random_test_batch_RNN(self, test_window) :
        #offset = int((self.DS.test_size - test_window) * np.random.rand(1))
        if self.msc == self.msn:
            self.msc = 0
        offset = allowed_trajectories[self.msc]
        self.msc += 1
        dynamic_range = range(offset, offset + test_window)
        batch_x, batch_y = self.getTestRNN(dynamic_range)
        return batch_x, batch_y

    def get_random_val_batch(self, test_window) :
        #offset = int((self.DS.val_size - test_window) * np.random.rand(1))
        if self.msc == self.msn:
            self.msc = 0
        offset = allowed_trajectories[self.msc]
        self.msc += 1
        dynamic_range = range(offset, offset + test_window)
        batch_x, batch_y = self.DS.getVal(dynamic_range)
        return batch_x, batch_y

    def get_random_val_batch_RNN(self, test_window) :
        #offset = int((self.DS.val_size - test_window) * np.random.rand(1))
        if self.msc == self.msn:
            self.msc = 0
        offset = allowed_trajectories[self.msc]
        self.msc += 1
        dynamic_range = range(offset, offset + test_window)
        batch_x, batch_y = self.DS.getValRNN(dynamic_range)
        return batch_x, batch_y




class LossSampler:
    def __init__(self, Dataset, multistep_number):
        self.DS = Dataset
        self.valPick = None
        self.testPick = None
        self.reset_val()
        self.reset_test()
        self.allowed_trajectories = None
        self.msc = 0
        self.msn = multistep_number

    def set_trajectories(self):
        self.allowed_trajectories = list(range(0,50,5000))

    def reset_test(self):
        self.testPick = np.ones(self.DS.test_size)

    def reset_val(self):
        self.valPick = np.ones(self.DS.test_size)

    def sample_4_test(self, batch_size):
        if np.sum(self.testPick) <= batch_size:
            self.reset_test()

        p = self.testPick/(np.sum(self.testPick))
        idxs = np.random.choice(self.DS.test_size,
                                  batch_size,
                                  p=p,
                                  replace = False)
        self.testPick[idxs] = 0
        return idxs

    def sample_4_val(self, batch_size):
        if np.sum(self.valPick) <= batch_size:
            self.reset_val()

        p = self.valPick/(np.sum(self.valPick))
        idxs = np.random.choice(self.DS.val_size,
                                  batch_size,
                                  p=p,
                                  replace = False)
        self.valPick[idxs] = 0
        return idxs

    def next_test_batch(self, batch_size):
        idxs = self.sample_4_test(batch_size)
        x, y = self.DS.getTest(idxs)
        return x, y


    def next_test_batch_RNN(self, batch_size):
        idxs = self.sample_4_test(batch_size)
        x, y = self.DS.getTestRNN(idxs)
        return x, y

    def next_val_batch(self, batch_size):
        idxs = self.sample_4_val(batch_size)
        x, y = self.DS.getVal(idxs)
        return x, y

    def next_val_batch_RNN(self, batch_size):
        idxs = self.sample_4_val(batch_size)
        x, y = self.DS.getValRNN(idxs)
        return x, y

    def val_batch(self, idxs):
        x, y = self.DS.getVal(idxs)
        return x, y


    def train_batch(self, idxs):
        x, y = self.DS.getTrain(idxs)
        return x, y

    def train_batch_RNN(self, idxs):
        x, y = self.DS.getTrainRNN(idxs)
        return x, y

    def next_train_batch_RNN(self, superbatch_size, batch_size, score, superbatch_indices):
        idxs,probs = self.sample_scored_batch(superbatch_size, batch_size, score, superbatch_indices)
        x, y = self.DS.getTrainRNN(idxs)
        return x, y

    def sample_superbatch(self, superbatch_size):
        """
        superbatch_size must be a multiple of batch_size
        """
        return np.random.choice(self.DS.train_size, superbatch_size)

    def sample_scored_batch(self, superbatch_size, batch_size, score, original_indices):
        p = score/np.sum(score)
        ids = np.random.choice(superbatch_size, batch_size, p=p[0])
        idxs = original_indices[ids]
        probs = p[0][ids]*superbatch_size
        return idxs, probs

    def get_random_test_batch(self, test_window) :
        #offset = int((self.DS.test_size - test_window) * np.random.rand(1))
        if self.msc == self.msn:
            self.msc = 0
        offset = allowed_trajectories[self.msc]
        self.msc += 1
        dynamic_range = range(offset, offset + test_window)
        batch_x, batch_y = self.DS.getTest(dynamic_range)
        return batch_x, batch_y

    def get_random_test_batch_RNN(self, test_window) :
        #offset = int((self.DS.test_size - test_window) * np.random.rand(1))
        if self.msc == self.msn:
            self.msc = 0
        offset = allowed_trajectories[self.msc]
        self.msc += 1
        dynamic_range = range(offset, offset + test_window)
        batch_x, batch_y = self.DS.getTestRNN(dynamic_range)
        return batch_x, batch_y

    def get_random_val_batch(self, test_window) :
        #offset = int((self.DS.val_size - test_window) * np.random.rand(1))
        if self.msc == self.msn:
            self.msc = 0
        offset = allowed_trajectories[self.msc]
        self.msc += 1
        dynamic_range = range(offset, offset + test_window)
        batch_x, batch_y = self.DS.getVal(dynamic_range)
        return batch_x, batch_y

    def get_random_val_batch_RNN(self, test_window) :
        offset = int((self.DS.val_size - test_window) * np.random.rand(1))
        dynamic_range = range(offset, offset + test_window)
        batch_x, batch_y = self.DS.getValRNN(dynamic_range)
        return batch_x, batch_y



