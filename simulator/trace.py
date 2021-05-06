import numpy as np
import scipy.io
import scipy.stats


def gen_irm_trace(sample_size, catalog_size, power_law_exp):
    # Generate a trace with power-law popularity (no timing is used so this is not precisely IRM)
    # M: sample size 
    # N: catalog size
    # s: power law exponent
    samples = np.random.uniform(low=0, high=1, size=sample_size)

    # Create power_law distribution
    # power_law = np.arange(1, catalog_size+1)**(-power_law_exp)
    # steps = np.cumsum(power_law / sum(power_law))

    steps = gen_power_law_dist(catalog_size, power_law_exp)

    # Find the index of the sample in the power-law distribution 
    trace = [np.where(sample <= steps)[0][0] for sample in samples] # [0][0] since the result of np.where is an tuple

    return np.array(trace)

def gen_dest_trace(sample_size, num_dest):
    return np.random.randint(low=0, high=num_dest, size=sample_size)


def gen_power_law_dist(catalog_size, power_law_exp):
    # Create power_law distribution
    power_law = np.arange(1, catalog_size+1)**(-power_law_exp)
    power_law_dist = np.cumsum(power_law / sum(power_law))
    return power_law_dist


def shot_noise_model_matlab(shot_duration, shot_rate, simulation_time=1000, par_shape=0.8, par_scale=1.6, par_loc=2):
    nof_shots = int(shot_rate*simulation_time)
    files_mean_requests = scipy.stats.genpareto.rvs(c=par_shape, scale=par_scale, loc=par_loc, size=nof_shots)
    files_shot_requests = scipy.stats.poisson.rvs(mu=files_mean_requests, size=nof_shots)

    total_samples = sum(files_shot_requests)

    trace_time = np.zeros(total_samples)
    trace_requests = np.zeros(total_samples)

    idx = 0
    for file_id, file_shot_requests in enumerate(files_shot_requests):
        trace_requests[idx:idx+file_shot_requests] = file_id
        
        shot_arrival = np.random.rand()*simulation_time
        requests_arrival = np.random.rand(file_shot_requests)*shot_duration
        trace_time[idx:idx+file_shot_requests] = shot_arrival + requests_arrival

        idx += file_shot_requests
    
    snm_trace = trace_requests[np.argsort(trace_time)].astype(int)
    return snm_trace



def shot_noise_model(max_shot_duration, shot_rate, simulation_time=1000, par_shape=0.8, par_scale=1.6, par_loc=2):
    nof_shots = int(shot_rate*simulation_time)
    files_mean_requests = scipy.stats.genpareto.rvs(c=par_shape, scale=par_scale, loc=par_loc, size=nof_shots)
    files_shot_requests = scipy.stats.poisson.rvs(mu=files_mean_requests, size=nof_shots)

    total_samples = sum(files_shot_requests)

    trace_time = np.zeros(total_samples)
    trace_requests = np.zeros(total_samples)

    idx = 0
    for file_id, file_shot_requests in enumerate(files_shot_requests):
        trace_requests[idx:idx+file_shot_requests] = file_id
        
        shot_arrival = np.random.rand()*simulation_time
        shot_duration = np.random.rand()*max_shot_duration

        requests_arrival = np.random.rand(file_shot_requests)*shot_duration
        trace_time[idx:idx+file_shot_requests] = shot_arrival + requests_arrival

        idx += file_shot_requests
    
    snm_trace = trace_requests[np.argsort(trace_time)].astype(int)
    return snm_trace


def get_yt_trace(yt_trace):
    # return load_mat_array(r"./traces/yt_trace.mat")['trace0']
    return load_mat_array(yt_trace)['trace0']
    

def load_mat_array(filename):
    return scipy.io.loadmat(filename, squeeze_me=True)


def get_trace_stats(trace: np.ndarray):
    sample_size = trace.size
    catalog_size = np.max(trace)+1
    return sample_size, catalog_size


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

    shot_noise_model_matlab(shot_duration=100, shot_rate=0.1, simulation_time=1000)

    