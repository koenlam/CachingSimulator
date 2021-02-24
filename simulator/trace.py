import numpy as np
import scipy.io


# M 1000000
# N 100
# s 0.8


def gen_irm_trace(sample_size, catalog_size, power_law_exp):
    # Generate a trace with power-law popularity (no timing is used so this is not precisely IRM)
    # M: sample size 
    # N: catalog size
    # s: power law exponent
    samples = np.random.uniform(low=0, high=1, size=sample_size)

    # Create power_law distribution
    power_law = np.arange(1, catalog_size+1)**(-power_law_exp)
    steps = np.cumsum(power_law / sum(power_law))

    # print(samples)

    # Find the index of the sample in the power-law distribution 
    # Note: in the highly unlikely case when the sample is exactly 1.0 this will crash
    trace = [np.where(sample < steps)[0][0] for sample in samples] # [0][0] since the result of np.where is an tuple


    return np.array(trace)

def get_yt_trace():
    return scipy.io.loadmat(r"./traces/yt_trace.mat")['trace0'][0]
    

if __name__ == "__main__":
    # t = gen_irm_trace(100, 100, 0.8)
    # print(t)
    yt = get_yt_trace()
    print(yt.size)