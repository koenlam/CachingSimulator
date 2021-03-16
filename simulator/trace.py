import numpy as np
import scipy.io
import scipy.stats
# from config import YT_TRACE


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

def gen_dest_trace(sample_size, num_dest):
    return np.random.randint(low=0, high=num_dest, size=sample_size)



def poisson_shot_model(shot_duration, pareto_pop_coeff, pareto_start, shot_rate, simulation_time=1000):
    # A_pareto=gprnd(0.8,1.6,2,nof_shots,1); %generate shot height (popularities)
    nof_shots = shot_rate*simulation_time
    t = scipy.stats.genpareto.rvs(c=0.8, scale=1.6, loc=2, size=(nof_shots, 1))
    print(t)



def get_yt_trace():
    # return load_mat_array(r"./traces/yt_trace.mat")['trace0']
    return load_mat_array(YT_TRACE)['trace0']
    

def load_mat_array(filename):
    return scipy.io.loadmat(filename, squeeze_me=True)


def parse_trace(mat_file : dict):
    trace = None

    # Naive way to parse the mat file
    for key in mat_file:
        if key[0] != "_":
            if trace is None:
                trace = mat_file[key]
            else:
                raise ValueError("Mat file contains multiple traces")

    sample_size = trace.size
    catalog_size = np.max(trace)
    trace = trace - 1 # Convert the Matlab indices starting at 1 to Python indices starting at 0
    return trace, catalog_size, sample_size
    

if __name__ == "__main__":
    # t = gen_irm_trace(100, 100, 0.8)
    # print(t)
    # yt = get_yt_trace()

    poisson_shot_model(shot_duration=100, pareto_pop_coeff=1.25, pareto_start=2, shot_rate=0.1, simulation_time=1000)

    