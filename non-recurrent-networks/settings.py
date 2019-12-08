import argparse
import os

class Settings:
    '''
    Stores all parameters set by the user + computes a few variables needed for training using them.
    '''
    def __init__(self):
        # Data and directories
        self.train_dir   = None 
        self.test_dir    = None 
        self.val_dir     = None 
        self.output_dir  = None 
        self.model_ckpt  = None
        self.tb_dir      = None 
        self.tb_log_name = None 
        # Tensorboard
        self.allow_tb = None
        # Cross-Validation
        self.use_X_val  = None 
        self.folds      = None
        self.test_ratio = None 
        self.val_ratio  = None 
        self.test_idx   = None 
        self.val_idx    = None 
        # Priorization settings
        self.priorization = None
        # PER settings
        self.alpha   = None 
        self.beta    = None 
        self.epsilon = None 
        # Grad settings
        self.superbatch_size = None
        # Data settings
        self.sequence_length   = None 
        self.forecast          = None
        self.trajectory_length = None 
        self.input_dim         = None 
        self.output_dim        = None 
        self.cmd_dim           = None 
        self.timestamp_idx     = None 
        self.continuity_idx    = None 
        # Reader
        self.reader_mode   = None
        # Training settings
        self.batch_size     = None 
        self.max_iterations = None 
        self.log_frequency  = None 
        self.learning_rate  = None 
        self.dropout        = None
        # Model
        self.model       = None
        self.restore     = None
        self.path_weight = None

        ### RUN ###
        self.run()

    def arg_parser(self):
        """
        Parses the arguments
        """
        parser = argparse.ArgumentParser()
        # Data and directories
        parser.add_argument('--train_data',type=str, help='path to the train data folder to be used')
        parser.add_argument('--test_data',type=str, required=False, help='path to the test data folder to be used')
        parser.add_argument('--val_data',type=str, required=False, help='path to the validation data folder to be used')
        parser.add_argument('--output', type=str, default='./result', help='path to the folder to save file to')
        parser.add_argument('--tb_dir', type=str, default='./tensorboard', help='path to the tensorboard directory')
        parser.add_argument('--tb_log_name', type=str, default='TOD', help='name of the log or TOD: Time Of Day, if TOD then the name will be automatically set to the time of the day')
        # Tensorboard
        parser.add_argument('--allow_tb', type=bool, default=True, help='path to the tensorboard directory')
        # Cross-Validation
        parser.add_argument('--use_cross_valid', type=bool, default=False, help='The ratio of data allocated for validation')
        parser.add_argument('--test_ratio', type=float, required=False, help='The ratio of data allocated to testing, if only providing one training set')
        parser.add_argument('--val_ratio', type=float, required=False, help='The ratio of data allocated for validation')
        parser.add_argument('--folds', type=int, required=False, help='The number of fold to perform during cross validation, standard value: 5.')
        parser.add_argument('--val_idx', type=int, required=False, help='The index of the validation fold')
        parser.add_argument('--test_idx', type=int, required=False, help='The index of the test fold')
        # Priorization settings
        parser.add_argument('--priorization', type=str, default='uniform', help='chose between the different type of optimization. uniform, PER or grad.')
        # PER settings
        parser.add_argument('--alpha', type=float, required=False, help='Alpha parameter as in PER by Schaul. (a value between 0 and 1)')
        parser.add_argument('--beta', type=float, required=False, help='Beta parameter as in PER by Schaul. (a value between 0 and 1)')
        parser.add_argument('--epsilon', type=float, required=False, help='Epsilon as in PER by Schaul.')
        parser.add_argument('--per_refresh_rate', type=int, required=False, help='Amount of steps after which the priorization weights have to be refreshed.')
        # Grad settings
        parser.add_argument('--superbatch_size', type=int, required=False, default='256', help='size of the super batch if using gradient upper-bound priorization scheme')
        # Data settings
        parser.add_argument('--max_sequence_size', type=int, default='12', help='number of points used to predict the outputs')
        parser.add_argument('--forecast', type=int, default='1', help='number of points to predict')
        parser.add_argument('--trajectory_length', type=int, default='20', help='size of the trajectory window')
        parser.add_argument('--input_dim', type=int, default='5', help='size of the input sample: using x,y,z coordinates as a data-point means input_dim = 3.')
        parser.add_argument('--output_dim', type=int, default='3', help='size of the sample to predict: predicting x, y, z velocities means output_dim = 3.')
        parser.add_argument('--timestamp_idx', type=int, required=False, help='Index of the timestamp if present in the data')
        parser.add_argument('--continuity_idx', type=int, required=False, help='Index of the continuity bit if present in the data')
        # Reader
        parser.add_argument('--reader_mode', type=str, default='classic', help='Chose the reader you want to use: classic, seq2seq, continuous_seq2seq')
        # Training settings 
        parser.add_argument('--batch_size', type=int, default='32', help='size of the batch')
        parser.add_argument('--max_iterations', type=int, default='10000', help='maximum number of iterations')
        parser.add_argument('--log_frequency', type=int, default='25', help='Loging frequency in batch.')
        parser.add_argument('--learning_rate', type=float, default='0.005', help='the learning rate')
        parser.add_argument('--dropout', type=float, default='0.75', help='the learning rate')
        # Model
        parser.add_argument('--model', type=str, help='the name of the model check code for available models')
        parser.add_argument('--restore', type=bool, required=False, default=False, help='use a pretrained model as weights for ours.')
        parser.add_argument('--weight_path', type=str, required=False, default='.', help='path to the weights')
        args = parser.parse_args()
        return args

    def assign_args(self, args):
        """
        Assigns the arguments
        """
        # Data and directories
        self.train_dir   = args.train_data
        self.test_dir    = args.test_data
        self.val_dir     = args.val_data
        self.output_dir  = args.output
        self.model_ckpt  = os.path.join(args.output,'model_ckpt')
        self.tb_dir      = args.tb_dir
        self.tb_log_name = os.path.join(self.tb_dir, self.tb_log_name)
        # Tensorboard
        self.allow_tb    = args.allow_tb
        # Cross-Validation
        self.use_X_val  = args.use_cross_valid
        self.test_ratio = args.test_ratio
        self.val_ratio  = args.val_ratio
        self.folds      = args.folds
        self.test_idx   = args.test_idx
        self.val_idx    = args.val_idx
        # Priorization settings
        self.priorization = args.priorization
        # PER settings
        self.alpha   = args.alpha
        self.beta    = args.beta
        self.epsilon = args.epsilon
        # Grad settings
        self.superbatch_size = args.superbatch_size
        # Data settings
        self.sequence_length   = args.max_sequence_size
        self.forecast          = args.forecast
        self.trajectory_length = args.trajectory_length
        self.input_dim         = args.input_dim
        self.output_dim        = args.output_dim
        self.cmd_dim           = args.input_dim - args.output_dim
        self.timestamp_idx     = args.timestamp_idx
        self.continuity_idx    = args.continuity_idx
        # Reader
        self.reader_mode   = args.reader_mode
        # Training settings
        self.batch_size     = args.batch_size
        self.max_iterations = args.max_iterations
        self.log_frequency  = args.log_frequency
        self.learning_rate  = args.learning_rate
        self.dropout        = args.dropout
        # Model
        self.model       = args.model
        self.restore     = args.restore
        self.path_weight = args.weight_path

    def generate_tensorboard_name(self, args):
        """
        Set the name of the tensorboard log directory for that run
        """
        if args.tb_log_name == 'TOD':
            from time import gmtime, strftime
            self.tb_log_name = strftime('%Y-%m-%d %H:%M:%S', gmtime())
            s = '_'
            self.tb_log_name = s.join(self.tb_log_name.split(' '))
            self.tb_log_name = s.join(self.tb_log_name.split('-'))
            self.tb_log_name = s.join(self.tb_log_name.split(':'))
        else:
            self.tb_log_name = args.tb_log_name

    def check_and_generate_directories(self):
        """
        Check that the specified directories match the requirements
        """
        # Check directories
        if not os.path.isdir(self.train_dir):
            raise ValueError('Cannot find directory ', self.train_dir)
        if (self.test_dir is not None) and (not os.path.isdir(self.test_dir)):
            raise ValueError('Cannot find directory ', self.test_dir)
        if (self.val_dir is not None) and (not os.path.isdir(self.val_dir)):
            raise ValueError('Cannot find directory ', self.val_dir)
        # Generates directories
        try:
            os.mkdir(self.output_dir)
        except:
            pass
        try:
            os.mkdir(self.tb_dir)
        except:
            pass
        try:
            os.mkdir(self.tb_log_name)
        except:
            pass

    def check_X_val(self):
        """
        Check that cross validation parameters make sense
        """
        if self.use_X_val:
            if self.folds < self.val_idx:
                raise ValueError("The validation index cannot be higher than the number of folds (indexing starts at 0 in python)")
            if self.folds < self.test_idx:
                raise ValueError("The test index cannot be higher than the number of folds (indexing starts at 0 in python)")

    def check_idxs(self):
        """
        Checks where the continuity_idx seats in the data, if it's after the timestamp_idx
        then when the timestamp are going to be removed the continuity idx will move of one
        point to the right. Correct the continuity_idx value accordingly
        """
        if self.continuity_idx:
            if self.continuity_idx > self.timestamp_idx:
                self.continuity_idx = self.continuity_idx - 1


    def run(self):
        """
        Loads the args checks path etc...
        """
        args = self.arg_parser()
        self.generate_tensorboard_name(args)
        self.assign_args(args)
        self.check_and_generate_directories()
        self.check_X_val()
        self.check_idxs()
