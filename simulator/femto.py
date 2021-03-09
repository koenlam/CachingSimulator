from .cache import *


class FemtoObj:
    def __init__(self, cache_size, catalog_size, caches_init, num_cache, cache_type, utilities, edges, num_user_locs):
        self.cache_size = cache_size
        self.catalog_size = catalog_size
        self.caches_init = caches_init
        self.num_cache = num_cache
        self.cache_type = cache_type
        self.utilities = utilities
        self.edges = edges
        self.num_user_locs = num_user_locs

        self.caches = [self.cache_type(self.cache_size, self.catalog_size, self.cache_init) for cache_init in self.caches_init]
        self.hits = []

    def reset(self):
        self.caches = [self.cache_type(self.cache_size, self.catalog_size, self.cache_init) for cache_init in self.caches_init]
        self.hits = []

class mLRU(FemtoObj):
    def __init__(self, cache_size, catalog_size, caches_init, num_cache, cache_type, utilities, edges, num_user_locs):
        super().__init__(cache_size, catalog_size, caches_init, num_cache, cache_type, utilities, edges, num_user_locs)
    

    def request(self, request, user_dest):
        local_utilities = self.utilities[user_dest,]
        local_edges = self.edges[user_dest,]

        z = np.zeros(self.num_cache)
        lazy_flag = 0

        for idx, utility in local_utilities:
            if local_edges[idx] == 1: # Cache reachable
                lazy_flag += 1
            
            z[idx] = np.min(len(np.where(self.caches[idx] == request)[0]), 1-np.sum(z))
        
        hit = np.multiply(z, local_utilities)

        if lazy_flag == 0:
            cache_options = np.where(local_edges > 0)[0]
            rnd_cache = random.choice(self.caches[cache_options])
            rnd_cache.request(request)
        elif lazy_flag > 0:
            for i, zi in enumerate(z):
                if zi > 0:
                    self.caches[i].request(request)
        return hit
                