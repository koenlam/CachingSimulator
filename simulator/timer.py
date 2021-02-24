import time

class Timer:
    def __init__(self):
        self._start_time = time.time()
    
    def tic(self):
        self._start_time = time.time()

    def toc(self):
        print("Elapsed time:", time.time()-self._start_time)


if __name__ == "__main__":
    t = Timer()
    print("Start")
    t.tic()
    time.sleep(3)
    t.toc()