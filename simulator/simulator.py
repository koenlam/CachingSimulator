import math
import numpy as np
import matplotlib.pyplot as plt

from .trace import gen_irm_trace, get_yt_trace
from .policy import LRU, LFU, OGD, gen_best_static, LRU_fast, LFU_fast, OGD_fast
from .cache import init_cache
from .timer import Timer

def simulate_cache(policy, trace, cache_init):
    cache = cache_init.copy()
    hits = []
    N = len(trace)
    percentage_mark = N // 10
    percentage_done = 0
    for i, request in enumerate(trace):
        # print(i)
        hit = policy(cache, request)
        hits.append(hit)

        # Print progress
        if i != 0 and (i % percentage_mark == 0 or i == N-1):
            percentage_done += 10
            print(f"{percentage_done}%")

    hitrate = np.cumsum(hits) / np.arange(1, len(trace)+1)
    return hitrate

def plot_comp(LRU, LFU, BH, OGD):
    t = np.arange(1, len(LRU)+1)
    plt.plot(t, LRU)
    plt.plot(t, LFU)
    plt.plot(t, BH)
    plt.plot(t, OGD)
    
    plt.xlabel("time")
    plt.ylabel("avg hits")
    plt.ylim([0, 0.8])

    plt.legend(("LRU", "LFU", "Best static", "OGD"))

    plt.show()

 

def simulate_trace(trace, cache_size, catalog_size, sample_size):
    timer = Timer()
    timer.tic()

    cache_init = init_cache(cache_size, catalog_size).tolist()

    print("LRU")
    hitrate_LRU = simulate_cache(LRU_fast, trace, cache_init)
    timer.toc()

    print("LFU")
    # timer.tic()
    LFU_fast.request_freq = np.ones(catalog_size).tolist()
    hitrate_LFU = simulate_cache(LFU_fast, trace, cache_init)
    timer.toc()


    print("Best static")
    # timer.tic()
    best_static_cache = gen_best_static(trace, cache_size, catalog_size)
    bs_func = lambda cache, request: 1 if request in cache else 0
    hitrate_BH = simulate_cache(bs_func, trace, best_static_cache)
    timer.toc()

    print("OGD")
    # timer.tic()
    hitrate_OGD = OGD(trace, cache_size, catalog_size, sample_size)
    
    # eta0 = math.sqrt(2*cache_size/sample_size)
    # OGD_func = lambda cache, request_file: OGD_fast(cache, request_file, cache_size, eta0)
    
    # OGD_cache_init = np.ones(catalog_size) * (cache_size / catalog_size)
    # OGD_cache_init = OGD_cache_init.tolist()
    
    
    # hitrate_OGD = simulate_cache(OGD_func, trace, OGD_cache_init)
    timer.toc()

    plot_comp(hitrate_LRU, hitrate_LFU, hitrate_BH, hitrate_OGD)