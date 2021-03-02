import cProfile
from simulator import *


if __name__ == "__main__":
    cache_size = 30
    catalog_size = 100
    # sample_size = 1000000
    sample_size = 100000
    power_law_exp = 0.8

    # irm_trace = gen_irm_trace(sample_size, catalog_size, power_law_exp)
    # simulate_trace(irm_trace, cache_size, catalog_size, sample_size)

    # catalog_size = 62538
    # cache_size = int(0.3*catalog_size)
    # yt_trace = get_yt_trace()-1
    # yt_trace = yt_trace.tolist()

    # simulate_trace(yt_trace, cache_size, catalog_size, sample_size)

    yt_trace = load_mat_array(r"traces/youtube_trace.mat")['trace']
    sample_size = yt_trace.size
    catalog_size = np.max(yt_trace)
    yt_trace -= 1
    # print(yt_trace)

    sample_size = 100000
    simulate_trace(yt_trace, cache_size, catalog_size, sample_size)
    


