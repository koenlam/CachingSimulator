import numpy as np

from collections import defaultdict


def convert2array(x):
    if isinstance(x, (defaultdict, dict)):
        x = np.array(x.keys())
    elif not isinstance(x, np.ndarray):
        x = np.array(x)
    return x