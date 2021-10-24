import numpy as np
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

from .cache import FTPL, init_cache, LRU, LFU, gen_best_static, CacheStatic, OGA
from .experts import ExpertCache, ExpertsCacheNeq
from .timer import Timer



def plot_comp(*caches):
    t = np.arange(1, len(caches[0].get_hitrate())+1)
    for cache in caches:
        plt.plot(t, cache.get_hitrate())
    
    plt.xlabel("time")
    plt.ylabel("avg hits")
#     plt.ylim([0, 0.8])
    plt.legend([cache.get_name() for cache in caches])

def simulate_trace(trace, cache_size, catalog_size, sample_size, cache_init=None, plot_hitrates=True):
    timer = Timer()
    timer.tic()

    if cache_init is None:
        cache_init = init_cache(cache_size, catalog_size)

    cache_LRU = LRU(cache_size, catalog_size, cache_init)
    cache_LFU = LFU(cache_size, catalog_size, cache_init)

    cache_BH = CacheStatic(cache_size, catalog_size, gen_best_static(trace, cache_size))
    cache_OGA = OGA(cache_size, catalog_size, sample_size)
    cache_EP_WM = ExpertCache(cache_size, catalog_size, cache_init, eps= 0.01, alg="WM")
    cache_EP_RWM = ExpertCache(cache_size, catalog_size, cache_init, eps= 0.01, alg="RWM")
    cache_FTPL = FTPL(cache_size, catalog_size, cache_init)

    OGA_init = lambda cache_size, catalog_size, cache_init: OGA(cache_size, catalog_size, sample_size)
    experts = (LRU, LFU, OGA_init, FTPL)

    cache_EP_neq = ExpertsCacheNeq(cache_size, catalog_size, cache_init, experts)

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

    print("Experts WM")
    cache_EP_WM.simulate(trace)
    timer.toc()

    print("Experts RWM")
    cache_EP_RWM.simulate(trace)
    timer.toc()

    print("OGD")
    cache_OGA.simulate(trace)
    timer.toc()

    print("FTPL")
    cache_FTPL.simulate(trace)
    timer.toc()

    print("Experts WM no equalization")
    cache_EP_neq.simulate(trace)
    timer.toc

    if plot_hitrates is True:
        print("Warning: plot hitrates has be deprecated")

   
    return cache_LRU, cache_LFU, cache_BH, cache_EP_WM, cache_EP_RWM, cache_OGA, cache_FTPL, cache_EP_neq


def simulate_caches_parallel(caches, trace):
    """ Simulates in parrallel using pathos multiprocessing
        Doesn't update the orginal cache
        Only returns the hits
    """
    import pathos.multiprocessing
    num_caches = len(caches)
    with tqdm(total=num_caches) as pbar:
        with pathos.multiprocessing.ProcessPool() as pool:
            futures = []
            for cache in caches:
                future = pool.apipe(cache.simulate, trace)
                futures.append(future)
            for future in futures: # Kind of hacky progress bar (It can happen that the next cache is already done simulating)
                future.wait()
                pbar.update(1)  
    return [future.get() for future in futures] # Return the hits of each cache