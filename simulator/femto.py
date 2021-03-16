from .cache import *


class FemtoObj:
    def __init__(self, cache_size, catalog_size, caches_init, cache_type, utilities, edges):
        self.cache_size = cache_size
        self.catalog_size = catalog_size
        self.caches_init = caches_init
        self.num_cache = edges.shape[1]
        self.cache_type = cache_type
        self.utilities = utilities
        self.edges = edges
        self.num_user_locs = edges.shape[0]

        # Sort the utilities on the highest utilities first
        self.utilities_sorted = []
        self.utilities_sorted_idx = []
        for utility in utilities:
            self.utilities_sorted.append(np.sort(utility)[::-1])
            self.utilities_sorted_idx.append(np.argsort(utility)[::-1])

        self.caches = [self.cache_type(
            self.cache_size, self.catalog_size, cache_init) for cache_init in self.caches_init]
        self.hits = []

    def get_cache(self):
        return [c.cache for c in self.caches]

    def get_hitrate(self):
        N = len(self.hits)
        return np.cumsum(np.sum(self.hits, axis=1)) / np.arange(1, N+1)

    def reset(self):
        self.caches = [self.cache_type(
            self.cache_size, self.catalog_size, cache_init) for cache_init in self.caches_init]
        self.hits = []


class mLRU(FemtoObj):
    def __init__(self, cache_size, catalog_size, caches_init, utilities, edges):
        super().__init__(cache_size, catalog_size, caches_init, LRU, utilities, edges)

    def request(self, request, dest):
        hit = np.zeros(self.num_user_locs)
        local_edges = self.edges[dest, ]
        z = np.zeros(self.num_cache, dtype=np.int)
        lazy_flag = 0

        for idx in self.utilities_sorted_idx[dest]:
            if local_edges[idx] == 1: # Cache reachable
                if request in self.caches[idx].get_cache():  
                    lazy_flag += 1

                z[idx] = min(np.where(self.caches[idx].cache == request)[
                            0].size, 1-np.sum(z))

            hit[dest] = np.dot(z, local_edges*self.utilities[dest, ])

        if lazy_flag == 0:
            cache_options = np.where(local_edges > 0)[0]
            # cache_choice = random.choice(cache_options)
            cache_choice = cache_options[0]
            self.caches[cache_choice].request(request)
        elif lazy_flag > 0:
            for i, zi in enumerate(z):
                if zi > 0:
                    self.caches[i].request(request)

        self.hits.append(hit)
        return hit


class LazyLRU(FemtoObj):
    def __init__(self, cache_size, catalog_size, caches_init, utilities, edges):
        super().__init__(cache_size, catalog_size, caches_init, LRU, utilities, edges)

    def request(self, request, dest):
        hit = np.zeros(self.num_user_locs)
        local_edges = self.edges[dest, ]
        z = np.zeros(self.num_cache, dtype=np.int)
        lazy_flag = 0

        for idx in self.utilities_sorted_idx[dest]:
            if local_edges[idx] == 1: # Cache reachable
                if request in self.caches[idx].get_cache():  
                    lazy_flag += 1

                z[idx] = min(np.where(self.caches[idx].cache == request)[
                            0].size, 1-np.sum(z))

        hit[dest] = np.dot(z, local_edges*self.utilities[dest, ])

        if lazy_flag == 0:
            cache_options = np.where(local_edges > 0)[0]
            cache_choice = random.choice(cache_options)
            self.caches[cache_choice].request(request)
        elif lazy_flag == 1:
            for i, zi in enumerate(z):
                if zi > 0:
                    self.caches[i].request(request)

        self.hits.append(hit)
        return hit
