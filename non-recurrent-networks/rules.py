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
                train_mode = 'classic'
            elif ((priorization == 'PER') or (priorization == 'per')):
                train_mode = 'per'
            elif ((priorization == 'GRAD') or (priorization == 'grad')):
                train_mode = 'grad'
            else:
                raise Exception('Unknown priorization mode. Currently supported modes are: uniform, PER, and GRAD.')
    elif sts.reader_mode == 'seq2seq':
        if name[0] in seq2pts:
            raise Exception('The selected model family : '+name[0]+' does not support '+sts.reader_mode+'.')
        else:
            if priorization == 'uniform':
                train_mode = 'seq2seq'
            elif ((priorization == 'PER') or (priorization == 'per')):
                train_mode = 'per_seq2seq'
            elif ((priorization == 'GRAD') or (priorization == 'grad')):
                train_mode = 'grad_seq2seq'
            else:
                raise Exception('Unknown priorization mode. Currently supported modes are: uniform, PER, and GRAD.')
    elif sts.reader_mode == 'continuous_seq2seq':
        if name[0] in seq2pts:
            raise Exception('The selected model family : '+name[0]+' does not support '+sts.reader_mode+'.')
        else:
            if priorization == 'uniform':
                train_mode = 'continuous_seq2seq'
            elif ((priorization == 'PER') or (priorization == 'per')):
                train_mode = 'per_continuous_seq2seq'
            elif ((priorization == 'GRAD') or (priorization == 'grad')):
                train_mode = 'grad_continuous_seq2seq'
            else:
                raise Exception('Unknown priorization mode. Currently supported modes are: uniform, PER, and GRAD.')
    else:
        raise Exception('Unknown reader mode. Currently supported modes are: classic, seq2seq, continuous_seq2seq')
    return train_mode

def check_if_completed(sts):
    

    
if __name__ == "__main__":
    settings = Settings()
    select_train_object(settings)

