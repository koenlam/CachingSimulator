import numpy as np

def test_LRU(verbose=False):
    cache_init = np.arange(0, 9)
    cache_LRU_miss = cache_init.copy()
    cache_LRU_hit = cache_init.copy()
    
    file_request_miss = 9
    LRU(cache_LRU_miss, file_request_miss)
    
    file_request_hit = 8
    LRU(cache_LRU_hit, file_request_hit)

    if verbose:
        print("LRU Test")
        print("Cache miss")
        print("Initial cache:", cache_init)
        print("File request:", file_request_miss)
        print("Cache result: ", cache_LRU_miss)

        print("\nCache hit")
        print("Initial cache:", cache_init)
        print("File request:", file_request_hit)
        print("Cache result: ", cache_LRU_hit)  


def test_LFU(verbose=False):
    cache_init = np.arange(0, 9)
    random.shuffle(cache_init)
    cache_LFU_miss = cache_init.copy()
    cache_LFU_hit = cache_init.copy()

    request_freq = np.arange(0, 20)
    request_freq[2] = 1 # To give files 1 and 2 the same frequency

    file_request_miss = 9
    LFU.request_freq = request_freq.copy()
    LFU(cache_LFU_miss, file_request_miss)

    file_request_hit = 5
    LFU.request_freq = request_freq.copy()
    LFU(cache_LFU_hit, file_request_hit)

    if verbose:
        print("LFU Test")
        print("Cache miss")
        print("Initial cache:", cache_init)
        print("File request:", file_request_miss)
        print("Cache result: ", cache_LFU_miss)

        print("\nCache hit")
        print("Initial cache:", cache_init)
        print("File request:", file_request_hit)
        print("Cache result: ", cache_LFU_hit)    


def test_best_static():
    from trace import gen_irm_trace

    trace = gen_irm_trace(sample_size=10, catalog_size=100, power_law_exp=0.8)
    print(trace)

    gen_best_static(trace, 10, 100)


def test_grad_proj():
    cache_size = 30
    catalog_size = 100
    sample_size = 100000
    request = 4
    eta0 = math.sqrt(2*cache_size/sample_size)

    cache = np.ones(catalog_size) * (cache_size / catalog_size)
    cache[request] += eta0
    cache_new = grad_proj(cache, cache_size, request)

    # np.set_printoptions(precision=10)
    print("Before:", cache)
    print("After:", cache_new)
