import cProfile
from simulator import *


def simulate_OGD(trace, cache_size, catalog_size, sample_size):
    eta0 = math.sqrt(2*cache_size/sample_size)
    OGD_func = lambda cache, request_file: OGD_fast(cache, request_file, cache_size, eta0)
    
    OGD_cache_init = np.ones(catalog_size) * (cache_size / catalog_size)
    OGD_cache_init = OGD_cache_init.tolist()

    hitrate_OGD = simulate_cache(OGD_func, trace, OGD_cache_init)



if __name__ == "__main__":
    cache_size = 30
    catalog_size = 100
    # sample_size = 1000000
    sample_size = 100000
    power_law_exp = 0.8

    irm_trace = gen_irm_trace(sample_size, catalog_size, power_law_exp)
    irm_trace = irm_trace.tolist()
    # simulate_trace(irm_trace, cache_size, catalog_size, sample_size)

    # test_func = lambda: simulate_trace(irm_trace, cache_size, catalog_size, sample_size)

    # cProfile.run("simulate_OGD(irm_trace, cache_size, catalog_size, sample_size)")
    # cProfile.run("OGD(irm_trace, cache_size, catalog_size, sample_size)")
    # catalog_size = 62538
    # yt_trace = get_yt_trace()-1
    # yt_trace = yt_trace.tolist()

    # print(sample_size)
    # print(len(yt_trace))
    # cProfile.run("OGD(yt_trace, cache_size, catalog_size, sample_size)")

    catalog_size = 62538
    cache_size = int(0.3*catalog_size)
    yt_trace = get_yt_trace()-1
    yt_trace = yt_trace.tolist()
    # print(yt_trace)
    # simulate_trace(yt_trace, cache_size, catalog_size, sample_size)
    cProfile.run("simulate_trace(yt_trace, cache_size, catalog_size, sample_size)")

    # cache = init_cache(cache_size, catalog_size).tolist()
    # request_freq = np.ones(catalog_size)

    # pr = cProfile.Profile()
    # t = Timer()
    # t.tic()
    # pr.enable()
    # for request_file in yt_trace:

    # pr.disable()
    # t.toc()
    # pr.print_stats(sort='time')