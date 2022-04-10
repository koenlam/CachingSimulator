# CachingSimulator
This simulator is built to evaluate various caching policies for my master's thesis. The results can be found [here](https://repository.tudelft.nl/islandora/object/uuid:68bbf523-2bbf-422b-9bd5-eac1e38e67d4). 

# Dependencies
 * Python 3.7+
 * Numpy
 * Matplotlib
 * Scipy
 * (Optional) Jupyter Notebook
 * (Optional) Tqdm
   * Used for displaying a loading bar
 * (Optional) Dill
   * Used to save the CacheObjects
 * (Optional) Pathos
   * Used for to multicore support in `simulate_caches_parallel()` otherwise use `simulate_caches()` for single core simulations
 
 
# Example
`
  python3 ./example.py
`
