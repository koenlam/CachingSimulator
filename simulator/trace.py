import random
import numpy as np
import scipy.io
import scipy.stats


def gen_irm_trace(sample_size, catalog_size, power_law_exp, shuffled=False):
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

    if shuffled is True:
        return shuffle_idx(trace, catalog_size)
    else:
        return np.array(trace)

def gen_dest_trace(sample_size, num_dest):
    return np.random.randint(low=0, high=num_dest, size=sample_size)


def gen_power_law_dist(catalog_size, power_law_exp):
    # Create power_law distribution
    power_law = np.arange(1, catalog_size+1, dtype=float)**(-power_law_exp)
    power_law_dist = np.cumsum(power_law / sum(power_law))
    return power_law_dist


def gen_uniform_trace(sample_size, catalog_size):
    return np.random.randint(low=0, high=catalog_size-1, size=sample_size)

def shuffle_idx(trace, catalog_size):
    """Shuffle the indices of the trace"""

    shuffled_idx = list(range(catalog_size))
    random.shuffle(shuffled_idx)

    split_idx = catalog_size // 2

    idx_p1 = shuffled_idx[:split_idx]
    idx_p2 = shuffled_idx[split_idx:]

    # When the catalog_size is uneven idx_p2 is larger than idx_p1
    # Here the "additional" elements of idx_p2 is added to idx_p1
    while len(idx_p2) > len(idx_p1):
        idx_p1.append(idx_p2[len(idx_p1)])

    swap_dict = dict(zip(idx_p1 + idx_p2, idx_p2 + idx_p1))

    trace_shuffled = [swap_dict[el] for el in trace]
    return trace_shuffled

def random_replacement_model(sample_size, catalog_size, power_law_exp, shuffled=False, replacement_rate=4):
    trace = gen_irm_trace(sample_size, catalog_size, power_law_exp)
    for i, _ in enumerate(trace):
        if random.random() <= 1/replacement_rate: # Swap according to the replacement rate
            i1 = random.randrange(catalog_size-5) + 5
            i2 = i1 - 5 

            i1_idx = np.where(trace[i:] == i1)[0]
            i2_idx = np.where(trace[i:] == i2)[0]

            trace[i+i1_idx] = i2
            trace[i+i2_idx] = i1

    return trace if shuffled is False else shuffle_idx(trace, catalog_size)

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
    


def get_movielens_trace():
    # MovieLens: https://grouplens.org/datasets/movielens/
    movielens_loc = r"./traces/MovieLens1M_ratings.dat"

    movielens_data = {"UserID": [], "MovieID": [], "Rating": [], "Timestamp": []} # UserID::MovieID::Rating::Timestamp

    try:
        with open(movielens_loc) as f:
            movielens_raw = f.read().strip().split("\n")

            for line in movielens_raw:
                UserID, MovieID, Rating, Timestamp = line.split("::")
                movielens_data["UserID"].append(UserID)
                movielens_data["MovieID"].append(MovieID)
                movielens_data["Rating"].append(Rating)
                movielens_data["Timestamp"].append(Timestamp)
    except FileNotFoundError:
        raise FileNotFoundError('MovieLens dataset not downloaded. \
Please run "download_movielens.sh" to download the dataset. \
Alternatively download the dataset manually and rename "ratings.dat" to "MovieLens1M_ratings.dat"') from None
    return np.array(movielens_data["MovieID"], dtype=int)



def combine_trace(*traces):
    return np.concatenate(traces)




if __name__ == "__main__":
  print(get_movielens_trace())