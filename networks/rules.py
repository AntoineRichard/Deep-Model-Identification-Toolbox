from settings import Settings
import train

def select_train_object(sts):
    seq2seq = ['RNN','LSTM','GRU','ATTN']
    seq2pts = ['MLP','CNN','MLPCPLX']
    name = sts.model.split('_')
    if sts.reader_mode == 'classic':
        if name[0] in seq2seq:
            raise Exception('The selected model family : '+name[0]+' does not support '+sts.reader_mode+'.')
        else:
            if priorization == 'uniform':
                train.Training_Uniform(sts)
            elif ((priorization == 'PER') or (priorization == 'per')):
                train.Training_PER(sts)
            elif ((priorization == 'GRAD') or (priorization == 'grad')):
                train.training_GRAD(sts)
            else:
                raise Exception('Unknown priorization mode. Currently supported modes are: uniform, PER, and GRAD.')
    elif sts.reader_mode == 'seq2seq':
        if name[0] in seq2pts:
            raise Exception('The selected model family : '+name[0]+' does not support '+sts.reader_mode+'.')
        else:
            if priorization == 'uniform':
                train.Training_Seq2Seq(sts)
            elif ((priorization == 'PER') or (priorization == 'per')):
                raise Exception('PER is not yet supported for seq2seq models.')
            elif ((priorization == 'GRAD') or (priorization == 'grad')):
                raise Exception('Gradient upperbound is not yet supported for seq2seq models.')
            else:
                raise Exception('Unknown priorization mode. Currently supported modes are: uniform, PER, and GRAD.')
    elif sts.reader_mode == 'continuous_seq2seq':
        if name[0] in seq2pts:
            raise Exception('The selected model family : '+name[0]+' does not support '+sts.reader_mode+'.')
        else:
            if priorization == 'uniform':
                train.Training_Continuous_Seq2Seq(sts)
            elif ((priorization == 'PER') or (priorization == 'per')):
                raise Exception('PER is not yet supported for seq2seq models.')
                train_mode = 'per_continuous_seq2seq'
            elif ((priorization == 'GRAD') or (priorization == 'grad')):
                raise Exception('Gradient upperbound is not yet supported for seq2seq models.')
                train_mode = 'grad_continuous_seq2seq'
            else:
                raise Exception('Unknown priorization mode. Currently supported modes are: uniform, PER, and GRAD.')
    else:
        raise Exception('Unknown reader mode. Currently supported modes are: classic, seq2seq, continuous_seq2seq')

def check_if_completed(sts):
    raise Exception('Not implemented')

    
if __name__ == "__main__":
    settings = Settings()
    #check_if_completed(settings)
    select_train_object(settings)

