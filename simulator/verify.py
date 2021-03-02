import matplotlib.pyplot as plt

from .simulator import simulate_trace
from .trace import load_mat_array, parse_trace


def compare_hitrate(cache, hitrate, title=""):
    cache_hitrate = cache.get_hitrate()
    plt.plot(cache_hitrate-hitrate)
    plt.xlabel("Time")
    plt.ylabel("Difference")
    plt.title(title)
    plt.show()

def verify_simulation(folder, relative_cache_size=0.3):
    # Read data
    mat_file = load_mat_array(f"{folder}/trace.mat")
    cache_init = load_mat_array(f"{folder}/init_cache.mat")['Initialize_cache']-1
    matlab_hitrates = load_mat_array(f"{folder}/hitrates.mat")

    # Parse data
    trace, catalog_size, sample_size = parse_trace(mat_file)

    # Simulate
    cache_size = int(relative_cache_size*catalog_size)
    cache_LRU, cache_LFU, cache_BH, cache_OGD = simulate_trace(trace, cache_size, catalog_size, sample_size, cache_init)

    # Plot the difference between the Python simulation and the Matlab simulation
    compare_hitrate(cache_LRU, matlab_hitrates['LRU_hitrate'], title="LRU")
    compare_hitrate(cache_LFU, matlab_hitrates['LFU_hitrate'], title="LFU")
    compare_hitrate(cache_BH, matlab_hitrates['BH_hitrate'], title="BH")
    compare_hitrate(cache_OGD, matlab_hitrates['OGD_hitrate'], title="OGD")

    return cache_LRU, cache_LFU, cache_BH, cache_OGD
    


