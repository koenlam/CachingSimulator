import random
import numpy as np
from simulator import Timer


def test_np_array_speed():
    print("Numpy array speed")
    trace = np.arange(1000000)
    cache = np.arange(30)
    timer = Timer()
    timer.tic()
    for t in trace:
        if t in cache:
            t = "something"
    timer.toc()


def test_list_speed():
    print("List speed")
    trace = list(range(1000000))
    # cache = np.arange(30)
    cache = list(range(30))
    timer = Timer()
    timer.tic()
    for t in trace:
        if t in cache:
            t = "something"
    timer.toc()

def test_np2list_speed():
    print("Numpy array speed")
    trace = np.arange(1000000)
    cache = np.arange(30)

    trace = trace.tolist()
    cache = cache.tolist()
    timer = Timer()
    timer.tic()
    for t in trace:
        if t in cache:
            t = "something"
    timer.toc()


def test_tolist():
    print("Tolist")
    trace = np.arange(1000000)
    timer = Timer()
    timer.tic()
    trace.tolist()
    timer.toc()



def test_np_argmin():
    t = np.ones(10).tolist() + np.arange(2, 10000000).tolist()
    random.shuffle(t)
    timer = Timer()

    print("Numpy")
    timer.tic()
    np_min_idx = np.argmin(t)
    print(np_min_idx)
    timer.toc()

    print("List")
    timer.tic()
    list_min = min(t)
    list_min_idx = [i for i, el in enumerate(t) if el == list_min]
    print(list_min_idx)
    timer.toc()


def test_sublist():
    timer = Timer()
    N = 100000
    subsize = 10000
    t = np.arange(N)
    subidx = np.array([random.randrange(N) for i in range(subsize)]).tolist()

    print("Numpy")
    timer.tic()
    r = t[subidx]
    timer.toc()

    print("List")
    t = list(t)
    subidx = list(subidx)
    timer.tic()
    r = [t[s] for s in subidx]
    timer.toc()
    



# test_np_array_speed()
# test_list_speed()
# test_tolist()
# test_np2list_speed()
# test_np_argmin()
test_sublist()