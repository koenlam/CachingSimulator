import numpy as np
import random


def init_cache(cache_size, catalog_size):
    """Initialize random cache"""
    # cache = list(range(1, cache_size+1))
    cache = np.arange(catalog_size)
    random.shuffle(cache)
    return cache[:cache_size]



if __name__ == "__main__":
    cache = init_cache(5, 10)
    print(cache)