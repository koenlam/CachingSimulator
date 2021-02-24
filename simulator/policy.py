import random
import math
import numpy as np


def LRU(cache, file_request):
    is_hit = False
    if file_request in cache: # Cache hit
        is_hit = True
        idx = cache.tolist().index(file_request)
        cache[0:idx+1] = np.roll(cache[0:idx+1], shift=1) # Bring file request to front
        return is_hit
    else: # Cache miss
        cache[:] = np.roll(cache, shift=1)
        cache[0] = file_request # Put file request in the cache
        return is_hit
        


def test_LRU():
    cache_init = np.arange(0, 9)
    cache_LRU_miss = cache_init.copy()
    cache_LRU_hit = cache_init.copy()
    
    file_request_miss = 9
    LRU(cache_LRU_miss, file_request_miss)
    
    file_request_hit = 8
    LRU(cache_LRU_hit, file_request_hit)

    print("LRU Test")
    print("Cache miss")
    print("Initial cache:", cache_init)
    print("File request:", file_request_miss)
    print("Cache result: ", cache_LRU_miss)

    print("\nCache hit")
    print("Initial cache:", cache_init)
    print("File request:", file_request_hit)
    print("Cache result: ", cache_LRU_hit)    


# def LFU(cache, request_file, request_freq):
#     is_hit = False
#     cache_new = cache.copy()
#     request_freq[request_file] += 1

#     if request_file in cache: # Cache hit
#         is_hit = True
#         return cache_new, is_hit
#     else: # Cache miss
#         cache_file_freq = request_freq[cache]
#         cache_file_low_freq_idx = random.choice([file_idx for file_idx, freq in enumerate(cache_file_freq) if freq == min(cache_file_freq)])

#         if request_freq[request_file] > request_freq[cache[cache_file_low_freq_idx]]:
#             cache_new[cache_file_low_freq_idx] = request_file
        
#         return cache_new, is_hit

def LFU(cache, request_file):
    is_hit = False
    LFU.request_freq[request_file] += 1

    if request_file in cache: # Cache hit
        is_hit = True
        return is_hit
    else: # Cache miss
        cache_file_freq = LFU.request_freq[cache]
        cache_file_low_freq_idx = random.choice([file_idx for file_idx, freq in enumerate(cache_file_freq) if freq == min(cache_file_freq)])

        if LFU.request_freq[request_file] > LFU.request_freq[cache[cache_file_low_freq_idx]]:
            cache[cache_file_low_freq_idx] = request_file
        
        return is_hit


def test_LFU():
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

    print("LFU Test")
    print("Cache miss")
    print("Initial cache:", cache_init)
    print("File request:", file_request_miss)
    print("Cache result: ", cache_LFU_miss)

    print("\nCache hit")
    print("Initial cache:", cache_init)
    print("File request:", file_request_hit)
    print("Cache result: ", cache_LFU_hit)    


def gen_best_static(trace, cache_size, catalog_size):
    empirical_freq = [(file, np.count_nonzero(trace == file)) for file in range(catalog_size)]
    empirical_freq.sort(key=lambda el: el[1], reverse=True)
    
    cache_best_static = list(map(lambda x: x[0], empirical_freq))[:cache_size]
    return cache_best_static

    

def test_best_static():
    from trace import gen_irm_trace

    trace = gen_irm_trace(sample_size=10, catalog_size=100, power_law_exp=0.8)
    print(trace)

    gen_best_static(trace, 10, 100)


def grad_proj(cache, cache_size, request):
    cache_new = cache.copy()
    while True:
        rho = ( sum(cache_new) - cache_size  ) / len( np.where(cache_new > 0)[0] )

        gzero_idx = np.where(cache_new > 0)[0]
        cache_new[gzero_idx] -= rho

        negative_values = np.where(cache_new < 0)[0]

        if len(negative_values) == 0:
            break
        else:
            gzero_idx = np.where(cache_new > 0)[0]

            cache_new[gzero_idx] = cache[gzero_idx]

            cache_new[negative_values] = 0

    if max(cache_new) > 1:
        cache_new = cache.copy()
        cache_new[request] = 0

        while True:
            rho = ( sum(cache_new) - cache_size +1 ) / len( np.where(cache_new > 0)[0] )

            gzero_idx = np.where(cache_new > 0)[0]
            cache_new[gzero_idx] -= rho

            negative_values = np.where(cache_new < 0)[0]

            if len(negative_values) == 0:
                cache_new[request] = 1
                break
            else:
                gzero_idx = np.where(cache_new > 0)[0]

                cache_new[gzero_idx] = cache[gzero_idx]

                cache_new[negative_values] = 0



    return cache_new


def OGD(trace, cache_size, catalog_size, sample_size):
    cache = np.ones(catalog_size) * (cache_size / catalog_size)
    eta0 = math.sqrt(2*cache_size/sample_size)

    hit_OGD = []
    for request in trace:
        hit_OGD.append(cache[request])
        cache[request] +=  eta0
        cache = grad_proj(cache, cache_size, request)

    hitrate = np.cumsum(hit_OGD) / np.arange(1, sample_size+1)
    return hitrate


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


if __name__ == "__main__":
    # test_LRU()
    test_LFU()
    # test_best_static()
    # test_grad_proj()