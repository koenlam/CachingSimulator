import numpy as np
import matplotlib.pyplot as plt

from tqdm.auto import tqdm


def plot_expert_choices(expert_cache, expert_names):
    t = np.arange(1, len(expert_cache.hits)+1)
    plt.plot(t, expert_cache.expert_choices, ".")
    plt.yticks(range(len(expert_names)), expert_names)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0), useMathText=True)
    plt.xlabel("Time")

def plot_comp(*caches, legend=True, legend_columns=1, legend_loc="best", ylabel="Hit Ratio"):
    t = np.arange(1, len(caches[0].get_hitrate())+1)
    for cache in caches:
        plt.plot(t, cache.get_hitrate())
    
    plt.xlabel("Time")
    plt.ylabel(ylabel)
#     plt.ylim([0, 0.8])
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0), useMathText=True)
    if legend:
        plt.legend([cache.get_name() for cache in caches], ncol=legend_columns, loc=legend_loc)


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

def simulate_caches_femto(caches, trace, destinations, separate_simulations=False):
    """ Simulates the caches with the trace
        Also updates the cache objects
        Option to simulate each cache seperately
    """
    if not separate_simulations:
        for request, dest in tqdm(zip(trace, destinations), total=len(trace)):
            for cache in caches:
                cache.request(request, dest) 
    else:
        for cache in caches:
            print(cache.name)
            for request, dest in tqdm(zip(trace, destinations), total=len(trace)):
                cache.request(request, dest)
            print()
    return caches


def simulate_caches_femto_parallel(caches, trace, destinations):
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
                future = pool.apipe(cache.simulate, trace, destinations)
                futures.append(future)
            for future in futures: # Kind of hacky progress bar (It can happen that the next cache is already done simulating)
                future.wait()
                pbar.update(1)  
    
    return [future.get() for future in futures]