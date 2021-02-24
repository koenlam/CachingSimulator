import numpy as np
import matplotlib.pyplot as plt

from .trace import gen_irm_trace, get_yt_trace
from .policy import LRU, LFU, OGD, gen_best_static
from .cache import init_cache
from .timer import Timer

def simulate_cache(policy, trace, cache_init):
    cache = cache_init.copy()
    hits = []
    N = len(trace)
    percentage_mark = N // 10
    percentage_done = 0
    for i, request in enumerate(trace):
        hit = policy(cache, request)
        hits.append(hit)

        # Print progress
        if i % percentage_mark == 0 or i == N:
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

 
# def simulate_irm():
#     cache_size = 30
#     catalog_size = 100
#     sample_size = 1000000
#     power_law_exp = 0.8

#     print("Generating trace")
#     trace = gen_irm_trace(sample_size, catalog_size, power_law_exp)
#     cache_init = init_cache(cache_size, catalog_size)

#     print("LRU")
#     hitrate_LRU = simulate_cache(LRU, trace, cache_init)

#     print("LFU")
#     LFU.request_freq = np.ones(catalog_size)
#     hitrate_LFU = simulate_cache(LFU, trace, cache_init)

#     print("Best static")
#     best_static_cache = gen_best_static(trace, cache_size, catalog_size)
#     bs_func = lambda cache, request: (cache, cache[request]) if request in cache else (cache, 0)
#     hitrate_BH = simulate_cache(bs_func, trace, best_static_cache)


#     print("OGD")
#     hitrate_OGD = OGD(trace, cache_size, catalog_size, sample_size)

#     plot_comp(hitrate_LRU, hitrate_LFU, hitrate_BH, hitrate_OGD)



def simulate_trace(trace, cache_size, catalog_size, sample_size):
    timer = Timer()

    cache_init = init_cache(cache_size, catalog_size)

    print("LRU")
    timer.tic()
    hitrate_LRU = simulate_cache(LRU, trace, cache_init)
    timer.toc()

    print("LFU")
    timer.tic()
    LFU.request_freq = np.ones(catalog_size)
    hitrate_LFU = simulate_cache(LFU, trace, cache_init)
    timer.toc()


    print("Best static")
    timer.tic()
    best_static_cache = gen_best_static(trace, cache_size, catalog_size)
    bs_func = lambda cache, request: 1 if request in cache else 0
    hitrate_BH = simulate_cache(bs_func, trace, best_static_cache)
    timer.toc()

    print("OGD")
    timer.tic()
    hitrate_OGD = OGD(trace, cache_size, catalog_size, sample_size)
    timer.toc()

    plot_comp(hitrate_LRU, hitrate_LFU, hitrate_BH, hitrate_OGD)


if __name__ == "__main__":
    cache_size = 30
    catalog_size = 100
    sample_size = 1000000
    power_law_exp = 0.8

    irm_trace = gen_irm_trace(sample_size, catalog_size, power_law_exp)
    simulate_trace(irm_trace, cache_size, catalog_size, sample_size)


    # catalog_size = 62538
    # cache_size = int(0.3*catalog_size)
    # yt_trace = get_yt_trace()
    # print(yt_trace)
    # simulate_trace(yt_trace, cache_size, catalog_size, sample_size)