import numpy as np
import matplotlib.pyplot as plt

from .cache import init_cache, LRU, LFU, gen_best_static, CacheStatic, OGD
from .timer import Timer



def plot_comp(*caches):
    t = np.arange(1, len(caches[0].get_hitrate())+1)
    for cache in caches:
        plt.plot(t, cache.get_hitrate())
    
    plt.xlabel("time")
    plt.ylabel("avg hits")
#     plt.ylim([0, 0.8])
    plt.legend([cache.get_name() for cache in caches])

def simulate_trace(trace, cache_size, catalog_size, sample_size, cache_init=None):
    timer = Timer()
    timer.tic()

    if cache_init is None:
        cache_init = init_cache(cache_size, catalog_size)

    cache_LRU = LRU(cache_size, catalog_size, cache_init)
    cache_LFU = LFU(cache_size, catalog_size, cache_init)

    cache_BH = CacheStatic(cache_size, catalog_size, gen_best_static(trace, cache_size))
    cache_OGD = OGD(cache_size, catalog_size, sample_size)

    trace = trace[:sample_size]
    print("LRU")
    cache_LRU.simulate(trace)
    timer.toc()

    print("LFU")
    cache_LFU.simulate(trace)
    timer.toc()

    print("Best static")
    cache_BH.simulate(trace)
    timer.toc()

    print("OGD")
    cache_OGD.simulate(trace)
    timer.toc()

    plot_comp(cache_LRU.get_hitrate(), cache_LFU.get_hitrate(), cache_BH.get_hitrate(), cache_OGD.get_hitrate())

    return cache_LRU, cache_LFU, cache_BH, cache_OGD