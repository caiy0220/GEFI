import numpy as np
import functools, warnings, yaml
import re, time, random, pickle
import torch
from pathlib import Path

from torch import manual_seed

""" Get device from tensor/model """
def get_device(x):
    if isinstance(x, torch.Tensor):
        d = x.device
    else:
        d = next(x.parameters()).device
    return d

def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return new_func

def get_file_path(root_path, model_name, dataset_name, expl_name, idx, filepath):
    filename = Path(filepath).stem
    return str(Path(root_path) / model_name.replace("/", "_") / dataset_name / expl_name / f"{idx}_{filename}.pkl")

def load_config(path):
    with open(path, 'r') as f:
        return yaml.full_load(f)

def set_seed(seed):
    random.seed(seed), np.random.seed(seed), manual_seed(seed)

'''
==================================================================
|                         Logging methods                        |
==================================================================
'''
LOG_LEVEL = 1
# 0: all log allowed
# 1: debug disabled
# 2: only warning and error
# 3: only error
LOG_PREF = ['DEBUG', 'INFO', 'WARN', 'ERR']
MAX_LINE_WIDTH = 80

def log(content='', lvl=1, end='\n'):
    if LOG_LEVEL > lvl:
        return

    pref = LOG_PREF[lvl] + '[' + time.strftime('%x,%X') + ']:'
    print(pref, end='\t')
    print(content, end=end)

def set_log_level(lvl):
    global LOG_LEVEL
    LOG_LEVEL = lvl

def heading(msg):
    remains = MAX_LINE_WIDTH - len(msg) - 2
    return '|' + ' ' * (remains // 2) + msg + ' ' * (remains // 2 + remains % 2) + '|'

# A simple use case for logging
def _logging_example():
    log('Current version of {}: 1.000.205'.format(__name__))    # Version info
    # Printing headline
    log('*' * MAX_LINE_WIDTH)
    log(heading('START INITIALIZATION'))
    log('*' * MAX_LINE_WIDTH)
    log('Some DEBUGGING information', lvl=0)

'''
==================================================================
|                          nested Pbar                           |
==================================================================
'''
WRAPPER_PBAR = False

class DescStr:
    def __init__(self):
        self._desc = ''

    def write(self, instr):
        self._desc += re.sub('\n|\x1b.*|\r', '', instr)

    def read(self):
        ret = self._desc
        self._desc = ''
        return ret

    def flush(self):
        pass

class Timer:
    def __init__(self):
        self.clocks = {}
    
    def tictoc(self, stamp, report=False):
        if stamp not in self.clocks:
            self.clocks[stamp] = time.time()
            diff = -1
        else:
            cur = time.time()
            diff = cur - self.clocks[stamp]
            if report:
                log(f'Time cost for {stamp}: {diff:.2f}')
            self.clocks[stamp] = cur
        return diff
 
def hide_axis(axs):
    for ax in axs.reshape(-1): ax.axis('off')


'''
==================================================================
|                   Log File Writing & Loading                   |
==================================================================
'''
def loadall(filename):
    ''' Reading explanations that are stored by entry in a .pkl file '''
    with open(filename, 'rb') as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break

def add_aopc2file(expl_name, fname, aopc, avg_cost=0, **kwargs):
    with open(fname, 'a') as f:
        num_samples = kwargs['num_samples']
        msg = f'{expl_name:<20}[{aopc:>5.2f}], #samples: {num_samples:>5}, #Avg cost: {avg_cost:5>.2f}s\n'
        f.write(msg)


'''
==================================================================
|              Some other functions for evaluation                |
==================================================================
'''
def random_ranking(num_features):
    ranking = list(range(num_features))
    random.shuffle(ranking)
    return ranking

# Sorting features (pixels/tokens) in descending order
def get_feature_ranking(vals, deletion=True):
    """ 
    Args:
        vals:       [torch.Tensor] Attribution/Interaction scores of pixels, in shape of (H, W)
        deletion:   [Bool] Ranking order, descending for deletion process
    -----
    Returns:
        Sorted vals in 1-D vector
        Feature Indices in 1-D vector
    """
    vals_flat = vals.flatten()
    res = vals_flat.sort(descending=deletion)    # (Values, Indices)
    # return list(zip(res[0].tolist(), res[1].tolist()))
    return res[0].tolist(), res[1].tolist()
