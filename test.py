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

test_np_array_speed()
test_list_speed()
test_tolist()
test_np2list_speed()
