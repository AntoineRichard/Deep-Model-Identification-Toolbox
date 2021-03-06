import os
from settings import Settings

def select_train_object(sts):
    seq2seq = ['RNN','LSTM','GRU','ATTNMP', 'ATTNMPMH', 'ATTNMPMHPHY']
    attn = ['ATTNMP','ATTNMPMH', 'ATTNMPMHPHY']
    rnn = ['RNN','LSTM','GRU']
    seq2pts = ['MLP','MLPPHY','MLPEN','CNN','MLPCPLX','ATTNSP']
    name = sts.model.split('_')
    if sts.reader_mode == 'classic':
        if name[0] in seq2seq:
            raise Exception('The selected model family : '+name[0]+' does not support '+sts.reader_mode+'.')
        else:
            if sts.priorization == 'uniform':
                train.Training_Uniform(sts)
            elif ((sts.priorization == 'PER') or (sts.priorization == 'per')):
                train.Training_PER(sts)
            elif ((sts.priorization == 'GRAD') or (sts.priorization == 'grad')):
                train.Training_GRAD(sts)
            else:
                raise Exception('Unknown priorization mode. Currently supported modes are: uniform, PER, and GRAD.')
    elif sts.reader_mode == 'seq2seq':
        if name[0] in seq2pts:
            raise Exception('The selected model family : '+name[0]+' does not support the '+sts.reader_mode+' reader mode.')
        elif name[0] in rnn:
            if sts.priorization == 'uniform':
                train.Training_RNN_Seq2Seq(sts)
            elif ((sts.priorization == 'PER') or (sts.priorization == 'per')):
                raise Exception('PER is not yet supported for seq2seq models.')
                train_mode = 'per_continuous_seq2seq'
            elif ((sts.priorization == 'GRAD') or (sts.priorization == 'grad')):
                raise Exception('Gradient upperbound is not yet supported for recurrent continuous seq2seq models.')
                train_mode = 'grad_continuous_seq2seq'
            else:
                raise Exception('Unknown priorization mode. Currently supported modes are: uniform, PER, and GRAD.')
        elif name[0] in attn:
            if sts.priorization == 'uniform':
                train.Training_Seq2Seq(sts)
            elif ((sts.priorization == 'PER') or (sts.priorization == 'per')):
                train.Training_Seq2Seq_PER(sts)
            elif ((sts.priorization == 'GRAD') or (sts.priorization == 'grad')):
                raise Exception('Gradient upperbound is not yet supported for seq2seq models.')
            else:
                raise Exception('Unknown priorization mode. Currently supported modes are: uniform, PER, and GRAD.')
        else:
            raise Exception('Unknown model.')
    elif sts.reader_mode == 'continuous_seq2seq':
        if name[0] in seq2pts:
            raise Exception('The selected model family : '+name[0]+' does not support '+sts.reader_mode+'.')
        elif name[0] in rnn:
            if sts.priorization == 'uniform':
                train.Training_RNN_Continuous_Seq2Seq(sts)
            elif ((sts.priorization == 'PER') or (sts.priorization == 'per')):
                raise Exception('PER is not yet supported for seq2seq models.')
                train_mode = 'per_continuous_seq2seq'
            elif ((sts.priorization == 'GRAD') or (sts.priorization == 'grad')):
                raise Exception('Gradient upperbound is not yet supported for recurrent continuous seq2seq models.')
                train_mode = 'grad_continuous_seq2seq'
            else:
                raise Exception('Unknown priorization mode. Currently supported modes are: uniform, PER, and GRAD.')
        elif name[0] in attn:
            if sts.priorization == 'uniform':
                train.Training_Continuous_Seq2Seq(sts)
            elif ((sts.priorization == 'PER') or (sts.priorization == 'per')):
                raise Exception('PER is not yet supported for seq2seq models.')
                train_mode = 'per_continuous_seq2seq'
            elif ((sts.priorization == 'GRAD') or (sts.priorization == 'grad')):
                raise Exception('Gradient upperbound is not yet supported for continuous seq2seq models.')
                train_mode = 'grad_continuous_seq2seq'
            else:
                raise Exception('Unknown priorization mode. Currently supported modes are: uniform, PER, and GRAD.')
        else:
            raise Exception('Unknown model.')
    elif sts.reader_mode == 'co_teaching':
        train.Training_CoTeaching(sts)
    else:
        raise Exception('Unknown reader mode. Currently supported modes are: classic, seq2seq, continuous_seq2seq')

def check_if_completed(sts):
    if os.path.exists(os.path.join(sts.output_dir,'statistics.txt')):
        exit(0)
    
if __name__ == "__main__":
    print("training")
    settings = Settings()
    check_if_completed(settings)
    import train
    print("still training")
    select_train_object(settings)
    print("done training")
