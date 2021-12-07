import numpy as np
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

from .cache import FTPL, init_cache, LRU, LFU, gen_best_static, CacheStatic, OGA
from .experts import ExpertCache, ExpertsCacheNeq
from .timer import Timer



def plot_expert_choices(expert_cache, expert_names):
    t = np.arange(1, len(expert_cache.hits)+1)
    plt.plot(t, expert_cache.expert_choices, ".")
    plt.yticks(range(len(expert_names)), expert_names)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0), useMathText=True)
    plt.xlabel("Time")

def plot_comp(*caches, legend=True, legend_columns=1, legend_loc="best"):
    t = np.arange(1, len(caches[0].get_hitrate())+1)
    for cache in caches:
        plt.plot(t, cache.get_hitrate())
    
    plt.xlabel("Time")
    plt.ylabel("Hit Ratio")
#     plt.ylim([0, 0.8])
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0), useMathText=True)
    if legend:
        plt.legend([cache.get_name() for cache in caches], ncol=legend_columns, loc=legend_loc)

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
    """ Simulates in parallel using pathos multiprocessing
        Doesn't update in-place unlike simulate_caches()
        Therefore, return assignment is needed
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
    
    return [future.get() for future in futures]


def simulate_caches(caches, trace, separate_simulations=False):
    """ Simulates the caches with the trace
        Also updates the cache objects
        Option to simulate each cache seperately
    """
    if not separate_simulations:
        for request in tqdm(trace, total=len(trace)):
            for cache in caches:
                cache.request(request) 
    else:
        for cache in caches:
            print(cache.name)
            for request in tqdm(trace, total=len(trace)):
                cache.request(request)
            print()
    return caches