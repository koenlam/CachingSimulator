import random
import math
import timeit
import numpy as np


def LRU(cache, file_request):
    is_hit = False
    if file_request in cache: # Cache hit
        is_hit = True
        # idx = cache.tolist().index(file_request)

        idx = np.where(cache == file_request)[0][0]

        cache[0:idx+1] = np.roll(cache[0:idx+1], shift=1) # Bring file request to front
        return is_hit
    else: # Cache miss
        cache[:] = np.roll(cache, shift=1)
        cache[0] = file_request # Put file request in the cache
        return is_hit


def LRU_fast(cache, file_request):
    is_hit = False
    if file_request in cache: # Cache hit
        is_hit = True
        idx = cache.index(file_request)
        # cache[0:idx+1] = np.roll(cache[0:idx+1], shift=1) # Bring file request to front
        cache[0:idx+1] = [cache[idx]] + cache[:idx]
        return is_hit
    else: # Cache miss
        # cache[:] = np.roll(cache, shift=1)
        # cache[0] = file_request # Put file request in the cache

        cache[:] = [file_request] + cache[:-1]
        return is_hit
        


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

def test_LRU_fast(verbose=False):
    cache_init = list(range(9))
    cache_LRU_miss = cache_init.copy()
    cache_LRU_hit = cache_init.copy()
    
    file_request_miss = 9
    LRU_fast(cache_LRU_miss, file_request_miss)
    
    file_request_hit = 8
    LRU_fast(cache_LRU_hit, file_request_hit)

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


def LFU_fast(cache, request_file):
    is_hit = False
    LFU_fast.request_freq[request_file] += 1

    if request_file in cache: # Cache hit
        is_hit = True
        return is_hit
    else: # Cache miss
        # cache_file_freq = [(file_idx, LFU.request_freq[c]) for file_idx, c in enumerate(cache)]
        # cache_file_low_freq_idx, cache_file_freq_min = min(cache_file_freq, key=lambda x: x[1])
        #### cache_file_low_freq_idx = random.choice([file_idx for file_idx, freq in enumerate(cache_file_freq) if freq == min(cache_file_freq)])
        # cache_file_low_freq_idx = random.choice([file_idx for file_idx, freq in enumerate(cache_file_freq) if freq == cache_file_freq_min])


        # cache_file_freq = [LFU_fast.request_freq[c] for c in cache]

        cache_file_freq = LFU_fast.request_freq[cache]

        cache_file_freq_min = np.min(cache_file_freq)
        # cache_file_low_freq_idx = random.choice([file_idx for file_idx, freq in enumerate(cache_file_freq) if freq == cache_file_freq_min])

        cache_file_low_freq_idx = np.random.choice(np.where(cache_file_freq == cache_file_freq_min)[0])


        if LFU_fast.request_freq[request_file] > cache_file_freq_min:
        # if LFU.request_freq[request_file] > cache_file_freq_min:
            cache[cache_file_low_freq_idx] = request_file
        return is_hit


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


# def gen_best_static(trace, cache_size, catalog_size):
#     empirical_freq = [(file, np.count_nonzero(trace == file)) for file in range(catalog_size)]
#     empirical_freq.sort(key=lambda el: el[1], reverse=True)
    
#     cache_best_static = list(map(lambda x: x[0], empirical_freq))[:cache_size]
#     return cache_best_static


def gen_best_static(trace, cache_size, catalog_size):
    if not isinstance(trace, np.ndarray):
        trace = np.array(trace)
    
    freq = np.bincount(trace)
    files = np.arange(catalog_size)
    
    empirical_freq = list(zip(files, freq[files]))
    empirical_freq.sort(key=lambda el: el[1], reverse=True)
    
    cache_best_static = list(map(lambda x: x[0], empirical_freq[:cache_size]))
    return cache_best_static




def test_best_static():
    from trace import gen_irm_trace

    trace = gen_irm_trace(sample_size=10, catalog_size=100, power_law_exp=0.8)
    print(trace)

    gen_best_static(trace, 10, 100)


def grad_proj_old(cache, cache_size, request):
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

def grad_proj(cache, cache_size, request):
    cache_new = cache.copy()
    while True:
        rho = ( np.sum(cache_new) - cache_size  ) / np.count_nonzero(cache_new)
        cache_new[cache_new > 0] = cache_new[cache_new > 0] - rho
        negative_values = np.where(cache_new < 0)[0]
        if len(negative_values) == 0:
            break
        else:
            values_to_reset = cache_new > 0
            cache_new[values_to_reset] = cache[values_to_reset]
            cache_new[negative_values] = 0

    if np.max(cache_new) > 1:
        cache_new = cache.copy()
        cache_new[request] = 0
        while True:
            rho = ( np.sum(cache_new) - cache_size +1 ) / np.count_nonzero(cache_new)
            cache_new[cache_new > 0] = cache_new[cache_new > 0] - rho
            negative_values = np.where(cache_new < 0)[0]
            if len(negative_values) == 0:
                cache_new[request] = 1
                break
            else:
                values_to_reset = cache_new > 0
                cache_new[values_to_reset] = cache[values_to_reset]
                cache_new[negative_values] = 0
    return cache_new


def OGD(trace, cache_size, catalog_size, sample_size):
    cache = np.ones(catalog_size) * (cache_size / catalog_size)
    eta0 = math.sqrt(2*cache_size/sample_size)

    hit_OGD = []
    N = len(trace)
    percentage_mark = N // 10
    percentage_done = 0
    for i, request in enumerate(trace):
        hit_OGD.append(cache[request])
        cache[request] = cache[request] + eta0
        cache = grad_proj(cache, cache_size, request)

        # Print progress
        if i != 0 and (i % percentage_mark == 0 or i == N-1):
            percentage_done += 10
            print(f"{percentage_done}%")

    hitrate = np.cumsum(hit_OGD) / np.arange(1, sample_size+1)
    return hitrate


def grad_proj_fast(cache, cache_size, request):
    # cache = cache.tolist()
    cache_new = cache.copy()
    while True:
        rho = (sum(cache_new) - cache_size) / len([c for c in cache_new if c > 0])
        
        for i, c in enumerate(cache_new):
            if c > 0:
                cache_new[i] -= rho

        negative_values_idx = [i for i, val in enumerate(cache_new) if val < 0]
        if len(negative_values_idx) == 0:
            break
        else:
            for i, c in enumerate(cache_new):
                if c > 0:
                    cache_new[i] = cache[i]
                elif c < 0:
                    cache_new[i] = 0

    if max(cache_new) > 1:
        cache_new = cache.copy()
        cache_new[request] = 0

        while True:
            rho = (sum(cache_new) - cache_size+1) / len([c for c in cache_new if c > 0])
            
            for i, c in enumerate(cache_new):
                if c > 0:
                    cache_new[i] -= rho

            negative_values_idx = [i for i, val in enumerate(cache_new) if val < 0]
            if len(negative_values_idx) == 0:
                cache_new[request] = 1
                break
            else:
                for i, c in enumerate(cache_new):
                    if c > 0:
                        cache_new[i] = cache[i]
                    elif c < 0:
                        cache_new[i] = 0
    return cache_new


def OGD_fast(cache, request_file, cache_size, eta0):
    hit = cache[request_file]
    cache[request_file] += eta0
    cache[:] = grad_proj_fast(cache, cache_size, request_file)
    return hit



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
    print(timeit.timeit(test_LRU, number=1000000))
    print(timeit.timeit(test_LRU_fast, number=1000000))
    # test_LFU()
    # test_best_static()
    # test_grad_proj()