import numpy as np

from collections import defaultdict


def convert2array(x):
    if isinstance(x, (defaultdict, dict)):
        x = np.array([key for key in x if x[key] > 0])
    elif not isinstance(x, np.ndarray):
        x = np.array(x)
    return x