import numpy as np
import random

from collections import defaultdict
from tqdm.auto import tqdm

from .tools import convert2array


def init_cache(cache_size, catalog_size):
    """Initialize random cache"""
    # cache = list(range(1, cache_size+1))
    cache = np.arange(catalog_size)
    random.shuffle(cache)
    return cache[:cache_size]

def gen_best_static(trace, cache_size):
    if not isinstance(trace, np.ndarray):
        trace = np.array(trace)

    files, count = np.unique(trace, return_counts=True)
    sort_idx = np.argsort(-count) # -count to get the most frequent first

    cache_best_static = files[sort_idx][:cache_size]

    # Convert from list of file into a default dict for faster lookup when simulating
    cache_dict = defaultdict(lambda: 0)
    for file in cache_best_static:
        cache_dict[file] = 1

    return cache_dict



class CacheObj:
    def __init__(self, cache_size, catalog_size, cache_init):
        self.cache_size = cache_size
        self.catalog_size = catalog_size
        self.cache_init = cache_init

        self.cache = cache_init.copy()
        self.hits = []

    def reset(self):
        self.cache = self.cache_init.copy()
        self.hits = []

    def get_cache(self):
        return self.cache
        
    def get_hitrate(self):
        N = len(self.hits)
        return np.cumsum(self.hits) / np.arange(1, N+1)

    def simulate(self, trace):
        N = len(trace)
        for i, request in tqdm(enumerate(trace), total=N):
            self.request(request)

    def get_name(self):
        return "CacheObj"


class CacheStatic(CacheObj):
    def __init__(self, cache_size, catalog_size, cache):
        super().__init__(cache_size, catalog_size, cache)

    def request(self, request):
        is_hit = self.cache[request]
        self.hits.append(is_hit)
        return is_hit
    
    def get_cache(self):
        if isinstance(self.cache, (defaultdict, dict)):
            return np.array([key for key in self.cache if self.cache[key] > 0])
        else:
            return super().get_cache()

    def get_name(self):
        return "CacheStatic"


class LRU(CacheObj):
    def __init__(self, cache_size, catalog_size, cache_init):
        super().__init__(cache_size, catalog_size, convert2array(cache_init))
    
    def request(self, request):
        is_hit = False
        if request in self.cache: # Cache hit
            is_hit = True
            # Bring file of request to the front of the cache
            idx = np.where(self.cache == request)[0][0]
            self.cache[0:idx+1] = np.roll(self.cache[0:idx+1], shift=1)
        else: # Cache miss
            # Put the file of the request to the front of the cache
            self.cache = np.roll(self.cache, shift=1)
            self.cache[0] = request

        self.hits.append(is_hit)
        return is_hit

    def get_name(self):
        return "LRU"

class LFU(CacheObj):
    def __init__(self, cache_size, catalog_size, cache_init):
        super().__init__(cache_size, catalog_size, convert2array(cache_init))
        self.file_freq = np.ones(self.catalog_size)

    def reset(self):
        super().reset()
        self.file_freq = np.ones(self.catalog_size)

    def request(self, request):
        is_hit = False
        self.file_freq[request] += 1

        if request in self.cache:  # Cache hit
            is_hit = True
        else:  # Cache miss
            cache_file_freq = self.file_freq[self.cache]
            cache_file_freq_min = np.min(cache_file_freq)

            if self.file_freq[request] > cache_file_freq_min:
                cache_file_freq_min_idx = np.random.choice(
                    np.where(cache_file_freq == cache_file_freq_min)[0])
                self.cache[cache_file_freq_min_idx] = request

        self.hits.append(is_hit)
        return is_hit

    def get_name(self):
        return "LFU"

class OGD(CacheObj):
    def __init__(self, cache_size, catalog_size, sample_size, cache_init=None, eta0=None):
        if cache_init is None:
            cache_init = np.ones(catalog_size) * (cache_size / catalog_size)
        super().__init__(cache_size, catalog_size, cache_init)

        if eta0 is None:
            self.eta0 = np.sqrt(2*cache_size/sample_size)
        else:
            self.eta0 = eta0

    def request(self, request, gradient=1):
        is_hit = self.cache[request]

        self.cache[request] = self.cache[request] + self.eta0*gradient
        self.cache = self._grad_proj(self.cache, self.cache_size, request)

        self.hits.append(is_hit)
        return is_hit

    @staticmethod
    def _grad_proj(cache, cache_size, request):
        cache_new = cache.copy()
        while True:
            rho = ( np.sum(cache_new) - cache_size  ) / np.count_nonzero(cache_new)
            cache_new[cache_new > 0] = cache_new[cache_new > 0] - rho
            negative_values = np.where(cache_new < 0)[0]
            if len(negative_values) == 0:
                break
            else:
                values_to_reset = cache_new > 0
                cache_new[values_to_reset] = cache[values_to_reset]
                cache_new[negative_values] = 0

        if np.max(cache_new) > 1:
            cache_new = cache.copy()
            cache_new[request] = 0
            while True:
                rho = ( np.sum(cache_new) - cache_size +1 ) / np.count_nonzero(cache_new)
                cache_new[cache_new > 0] = cache_new[cache_new > 0] - rho
                negative_values = np.where(cache_new < 0)[0]
                if len(negative_values) == 0:
                    cache_new[request] = 1
                    break
                else:
                    values_to_reset = cache_new > 0
                    cache_new[values_to_reset] = cache[values_to_reset]
                    cache_new[negative_values] = 0
        return cache_new

    def get_name(self):
        return "OGD"

