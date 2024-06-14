import json
from collections import OrderedDict
from itertools import repeat
from pathlib import Path

import pandas as pd
import torch

try:
    from ruamel.yaml import YAML
    yaml = YAML()
except ImportError:
    pass

FILE_TYPE = 'yaml'

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_yaml(fname):
    global yaml
    fname = Path(fname)
    return yaml.load(fname)

def write_yaml(content, fname):
    global yaml
    fname = Path(fname)
    with fname.open('wt') as handle:
        yaml.dump(content, handle)

def read_config(fname):
    global FILE_TYPE
    fname = Path(fname)
    if fname.suffix.lower().endswith('json'):
        FILE_TYPE = 'json'
        return read_json(fname)
    elif fname.suffix.lower().endswith('yaml'):
        FILE_TYPE = 'yaml'
        return read_yaml(fname)
    else:
        raise ValueError("Unrecognized file type.")

def write_config(fname):
    fname = Path(fname)
    if FILE_TYPE == 'json':
        write_json(fname)
    elif FILE_TYPE == 'yaml':
        write_yaml(fname)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.loc[key, 'total'] += value * n
        self._data.loc[key, 'counts'] += n
        self._data.loc[key, 'average'] = self._data.loc[key, 'total'] / self._data.loc[key, 'counts']

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
