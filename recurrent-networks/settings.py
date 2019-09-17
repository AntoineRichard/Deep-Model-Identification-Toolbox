import argparse

class Settings:
    '''
    Stores all parameters set by the user + computes a few variables needed for training using them.
    '''
    def __init__(self):
        ### GENERAL PARAMETERS ###
        self.dataset_dir = None
        self.output_dir = None
        self.train_length = None
        self.test_length = None
        self.val_length = None
        self.sequence_length = None
        self.test_ratio = None
        self.val_ratio = None
        self.log_frequency = None
        ### NETWORK TUNNING ###
        self.input_dim = None
        self.output_shape = None
        self.state_size = None
        self.rnn_model = None
        self.num_layers = None
        self.priorization = None
        ### Training Variables ###
        self.learning_rate = None
        self.batch_size = None
        self.num_epochs = None
        self.num_batches_train = None
        self.num_batches_val = None  
        self.num_batches_test = None

        ### RUN ###
        self.run()

    def arg_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--train_data',type=str, help='path to the train data folder to be used')
        parser.add_argument('--test_data',type=str, help='path to the test data folder to be used')
        parser.add_argument('--val_data',type=str, help='path to the validation data folder to be used')
        parser.add_argument('--output', type=str, default='./result', help='path to the folder to save file to')
        parser.add_argument('--test_ratio', type=float, default='0.2', help='The ratio of data allocated to testing')
        parser.add_argument('--val_ratio', type=float, default='0.1', help='The ratio of data allocated for validation')
        parser.add_argument('--batch_size', type=int, default='16', help='size of the batch')
        parser.add_argument('--state_size', type=int, default='32', help='size of the batch')
        parser.add_argument('--max_epochs', type=int, default='10', help='maximum number of epochs')
        parser.add_argument('--max_trajectories', type=int, default='49', help='maximum number of validation trajectories')
        parser.add_argument('--window_size', type=int, default='20', help='size of the trajectory window')
        parser.add_argument('--max_sequence_size', type=int, default='16', help='number of points used to predict the outputs')
        parser.add_argument('--input_dim', type=int, default='5', help='size of the input sample: using x,y,z coordinates as a data-point means input_dim = 3.')
        parser.add_argument('--output_dim', type=int, default='3', help='size of the sample to predict: prediction x y z positions means output_dim = 3.')
        parser.add_argument('--layer_number', type=int, default='2', help='number of layer in the RNN.')
        parser.add_argument('--log_frequency', type=int, default='50', help='Loging frequency in batch.')
        parser.add_argument('--learning_rate', type=float, default='0.005', help='the learning rate')
        parser.add_argument('--rnn_model', type=str, default='lstm-basic', help='rnn mode check code for available models')
        parser.add_argument('--optimization', type=str, default='none', help='chose between the different type of optimization. Check the code for available types.')
        parser.add_argument('--tb_dir', type=str, default='./tensorboard', help='path to the tensorboard directory')
        parser.add_argument('--tb_log_name', type=str, default='TOD', help='name of the log or TOD: Time Of Day, if TOD then the name will be automatically set to the time of the day')
        parser.add_argument('--allow_tb', type=bool, default=True, help='path to the tensorboard directory')
        parser.add_argument('--weighting_mode', type=str, default='PER', help='Mode used for weighting, PER, basic or custom.')
        parser.add_argument('--alpha', type=float, default='0.8', help='alpha parameter as in PER by Schaul')
        parser.add_argument('--beta', type=float, default='0.3', help='beta parameter as in PER by Schaul.')
        parser.add_argument('--epsilon', type=float, default='0.0000001', help='epsilon as in PER by Schaul.')
        args = parser.parse_args()
        return args

    def assign_args(self):
        ### GENERAL PARAMETERS ###
        args = self.arg_parser()
        self.output_dir = args.output
        self.sequence_length = args.max_sequence_size
        self.train_dir = args.train_data
        self.test_dir = args.test_data
        self.val_dir = args.val_data
        self.test_ratio = args.test_ratio
        self.val_ratio = args.val_ratio
        self.tb_dir = args.tb_dir
        self.tb_log_name = args.tb_log_name
        self.allow_tb = args.allow_tb
        ### NETWORK TUNNING ###
        self.input_dim = args.input_dim
        self.output_dim = args.output_dim
        self.cmd_dim = args.input_dim - args.output_dim
        self.state_size = args.state_size
        self.rnn_model = args.rnn_model
        self.priorization = args.optimization
        self.num_layers = args.layer_number
        self.log_frequency = args.log_frequency
        ### PRIORIZATION TUNNING ###
        self.weighting_mode = args.weighting_mode
        self.alpha = args.alpha
        self.beta = args.beta
        self.epsilon = args.epsilon
        ### Training Variables ###
        self.learning_rate = args.learning_rate
        self.num_epochs = args.max_epochs
        self.max_trajectories = args.max_trajectories
        self.window_size = args.window_size
        self.batch_size = args.batch_size

    def data_parameters(self, data):
        self.train_length = data.train.shape[0]
        self.test_length = data.test.shape[0]
        self.val_length = data.val.shape[0]
        self.num_batches_train = self.train_length//self.batch_size//self.sequence_length - 3
        self.num_batches_val = self.val_length//self.batch_size//self.sequence_length - 3
        self.num_batches_test = self.test_length//self.batch_size//self.sequence_length - 3

    def generate_tensorboard_name(self):
        if self.tb_log_name == 'TOD':
            from time import gmtime, strftime
            self.tb_log_name = strftime("%Y-%m-%d %H:%M:%S", gmtime())

    def run(self):
        self.assign_args()
        self.generate_tensorboard_name()
