import os
import re

import numpy as np

from collections import defaultdict


def convert2array(x):
    if isinstance(x, (defaultdict, dict)):
        x = np.array([key for key in x if x[key] > 0])
    elif not isinstance(x, np.ndarray):
        x = np.array(x)
    return x


def save_caches(caches, pathname, split=True):
    import dill
    if split:
        for i, cache in enumerate(caches):
            with open(f"{pathname}-{i}", "wb") as f:
                dill.dump(cache, f)
    else:
          with open(f"{pathname}-0", "wb") as f:
                dill.dump(caches, f)
        
def load_caches(pathname):
    import dill
    tmp = []
    dirname = dirname = os.path.dirname(os.path.abspath(pathname))
    filename = pathname.split("/")[-1]
    for file in sorted(os.listdir(dirname)):
        if re.findall(f"{filename}-", file):
            with open(f"{dirname}/{file}", "rb") as f:
                tmp.append(dill.load(f))
    return tmp