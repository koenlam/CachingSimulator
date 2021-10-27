import numpy as np
import random

from collections import defaultdict
from tqdm.auto import tqdm

from .tools import convert2array

DEFAULT_SWITCHING_COST = 0.5

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
    def __init__(self, cache_size, catalog_size, cache_init, switching_cost=DEFAULT_SWITCHING_COST, metric="hit"):
        self.cache_size = cache_size
        self.catalog_size = catalog_size
        self.cache_init = cache_init
        self.switching_cost = switching_cost
        self.metric = metric

        self.name = "CacheObj"


    def reset(self):
        self.cache = self.cache_init.copy()
        self.cache_prev = self.cache.copy()
        self.hits = []
        self.reward_sw = [] # Reward + cache switching cost
        self.get_cache_diff = self.get_cache_diff_whole_vec


    def update_perf_metrics(self, is_hit, cache_switching_cost="default"):

        self.hits.append(is_hit)

        if self.metric == "switch": # Calculating cache switching cost is slow
            if cache_switching_cost == "default":
                cache_switching_cost = self.get_cache_switching_cost()
            self.reward_sw.append(float(is_hit) - cache_switching_cost)
        elif self.metric == "hit":
            pass
        else:
            raise ValueError(f"Unknown metric {self.metric}")
        self.cache_prev = self.cache.copy()


    def get_cache_diff_file_idx(self, cache, cache_prev):
        """Get the difference between caches when the cache consists of file IDS"""
        in_curr_not_prev = np.setdiff1d(cache, cache_prev)
        in_prev_not_curr = np.setdiff1d(cache_prev, cache)

        in_curr_not_prev_vec = np.ones(in_curr_not_prev.shape)
        in_prev_not_curr_vec = -np.ones(in_prev_not_curr.shape)

        return np.concatenate((in_curr_not_prev_vec, in_prev_not_curr_vec))


    def get_cache_diff_whole_vec(self, cache, cache_prev):
        """Get the difference between caches when the files in the caches are encoded with 1 in cache and 0 not in cache """
        return cache - cache_prev # Vector with 1 for elements in cache and not in cache_prev -1 for elements in cache_prev and not in cache


    def get(self, request):
        """Return whether or not a request is in the cache. Doesn't update the cache"""
        return request in self.cache

    def get_cache_switching_cost(self, cache=None, cache_prev=None, ord=1):
        if cache is None:
            cache = self.cache
        if cache_prev is None:
            cache_prev = self.cache_prev
        # 1-norm diff between cache and prev cache
        # switching cost /  2 due to the difference being counted twice
        return (self.switching_cost / 2) * np.linalg.norm(self.get_cache_diff(cache, cache_prev), ord=ord) 
        
    def get_cache(self):
        return self.cache
        
    def get_hitrate(self):
        N = len(self.hits)
        return np.cumsum(self.hits) / np.arange(1, N+1)

    def simulate(self, trace):
        N = len(trace)
        for i, request in tqdm(enumerate(trace), total=N):
            self.request(request)
        return self.hits

    def get_name(self):
        return self.name

    def get_ranking(self):
        """ Returns the ranking of the cache contents    
        """ 
        raise NotImplementedError("get_ranking not implemented")


class CacheStatic(CacheObj):
    def __init__(self, cache_size, catalog_size, cache):
        super().__init__(cache_size, catalog_size, cache)
        self.name = "CacheStatic"
        self.reset()

    def get_ranking(self):
        ranking = np.zeros(self.catalog_size)
        val = np.sum(np.arange(1, self.cache_size+1)) / self.cache_size 
        for c in self.cache:
            ranking[c] =  val 

        assert np.sum(ranking) ==  np.sum(np.arange(1, self.cache_size+1))
        return ranking # Ranks each content in the cache as equally important and sum(ranking)  = sum(1 + 2 + ... + self.cache_size)

    def request(self, request):
        is_hit = self.cache[request]
        self.update_perf_metrics(is_hit, cache_switching_cost=0)
        return is_hit
    
    def get_cache(self):
        if isinstance(self.cache, (defaultdict, dict)):
            return np.array([key for key in self.cache if self.cache[key] > 0])
        else:
            return super().get_cache()
    


class LRU(CacheObj):
    def __init__(self, cache_size, catalog_size, cache_init):
        super().__init__(cache_size, catalog_size, convert2array(cache_init))
        self.name = "LRU"
        self.reset()

    def reset(self):
        super().reset()
        self.get_cache_diff = self.get_cache_diff_file_idx

    def get_ranking(self):
        ranking = np.zeros(self.catalog_size)

        for i, c in enumerate(self.cache):
            ranking[c] =  (self.cache_size - i)
        assert np.sum(ranking) ==  np.sum(np.arange(1, self.cache_size+1))
        return ranking
    
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

        self.update_perf_metrics(is_hit)
        return is_hit


class LFU(CacheObj):
    def __init__(self, cache_size, catalog_size, cache_init):
        super().__init__(cache_size, catalog_size, convert2array(cache_init))
        self.name = "LFU"
        self.reset()

    def reset(self):
        super().reset()
        self.file_freq = np.ones(self.catalog_size)
        self.get_cache_diff = self.get_cache_diff_file_idx

    def get_ranking(self):
        ranking = np.zeros(self.catalog_size)

        cache_file_freq = self.file_freq[self.cache]
        cache_sorted = self.cache[np.argsort(cache_file_freq)] # Most important files at the end

        for i, c in enumerate(cache_sorted):
            ranking[c] = i+1 # 1 for the least important and self.cache_size for the most important
        
        assert np.sum(ranking) ==  np.sum(np.arange(1, self.cache_size+1))
        return ranking



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

        self.update_perf_metrics(is_hit)
        return is_hit


class OGA(CacheObj):
    def __init__(self, cache_size, catalog_size, sample_size, cache_init=None, eta0=None):
        # OGA uses a different cache representation than the other cache objects
        # Other cache objects use an list of objects id's
        # OGA uses an array of the whole catalog size with 1 if it is in the cache, 0 if it is not, and e.g. 0.5 if it is partially in the cache
        # Therefore, the cache init has to be converted 
        if cache_init is None:
            OGA_cache_init = np.ones(catalog_size) * (cache_size / catalog_size)
        elif len(cache_init) == cache_size: 
            OGA_cache_init = np.zeros(catalog_size)
            OGA_cache_init[cache_init] = 1
        else:
            OGA_cache_init = cache_init

        super().__init__(cache_size, catalog_size, OGA_cache_init)

        if eta0 is None:
            self.eta0 = np.sqrt(2*cache_size/sample_size)
        else:
            self.eta0 = eta0
        self.name = "OGA"
        self.reset()

    def get(self, request):
        return self.cache[request]

    def request(self, request, gradient=1):
        is_hit = self.cache[request]

        self.cache[request] = self.cache[request] + self.eta0*gradient
        self.cache = self._grad_proj(self.cache, self.cache_size, request)

        self.update_perf_metrics(is_hit)
        return is_hit

    @staticmethod
    def _grad_proj(cache, cache_size, request):
        cache_new = cache.copy()
        while True:
            # Calculate the how much each cache value has to be lowered such that the size of the cache the same as cache_size
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


class DiscreteOGA(CacheObj):
    def __init__(self, cache_size, catalog_size, sample_size, cache_init, eta0=None):
        super().__init__(cache_size, catalog_size, cache_init)
        self.OGA = OGA(cache_size, catalog_size, sample_size, cache_init=cache_init, eta0=eta0)

        self.name = "Discrete OGA"
        self.reset()

    def get_ranking(self):
        ranking = np.zeros(self.catalog_size)

        for i, c in enumerate(self.cache):
            ranking[c] = i+1 # 1 for the least important and self.cache_size for the most important
        
        assert np.sum(ranking) ==  np.sum(np.arange(1, self.cache_size+1))
        return ranking

    def request(self, request, gradient=1):
        is_hit = True if request in self.cache else False

        # Update OGA
        self.OGA.request(request, gradient)

        if not is_hit: # Cache miss
            OGA_file_importance = self.OGA.cache
            self.cache = np.argsort(OGA_file_importance)[-self.cache_size:]
        self.update_perf_metrics(is_hit)
        return is_hit




class BestDynamicCache(CacheObj):
    def __init__(self, cache_size, catalog_size):
        super().__init__(cache_size, catalog_size, np.zeros(cache_size))
        self.file_freq = np.ones(self.catalog_size)
        self.name = "Best Dynamic"
        self.reset()

    def request(self, request):
        # Update cache
        self.file_freq[request] += 1
        self.cache = np.argsort(self.file_freq)[-self.cache_size:]

        is_hit = True if request in self.cache else False
        self.update_perf_metrics(is_hit)
        return is_hit


class FTPL(CacheObj):
    def __init__(self, cache_size, catalog_size, cache_init, eta=1.0):
        super().__init__(cache_size, catalog_size, cache_init)
        self.file_freq = np.ones(self.catalog_size)
        self.eta = eta
        self.name = "FTPL"
        self.reset()

    def reset(self):
        super().reset()
        self.get_cache_diff = self.get_cache_diff_file_idx

    def request(self, request):
        is_hit = True if request in self.cache else False

        # Update cache
        self.file_freq[request] += 1
        y = self.file_freq + self.eta*np.random.randn(self.file_freq.size)
        self.cache = np.argsort(y)[-self.cache_size:]
        self.update_perf_metrics(is_hit)
        return is_hit
