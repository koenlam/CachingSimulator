
from scipy import optimize

from simulator.experts2 import RankingExperts

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

        self.caches = []
        self.hits = []

    def init_caches(self):
        self.caches = [self.cache_type(
            self.cache_size, self.catalog_size, cache_init) for cache_init in self.caches_init]


    def get_cache(self):
        return [c.cache for c in self.caches]

    def get_hitrate(self):
        N = len(self.hits)
        return np.cumsum(np.sum(self.hits, axis=1)) / np.arange(1, N+1)

    def reset(self):
        self.init_caches()
        self.hits = []

    def get_name(self):
        return "FemtoObj"


class mLRU(FemtoObj):
    def __init__(self, cache_size, catalog_size, caches_init, utilities, edges):
        super().__init__(cache_size, catalog_size, caches_init, LRU, utilities, edges)
        self.reset()

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
            # cache_choice = cache_options[0] # Used to compare with the Matlab version
            self.caches[cache_choice].request(request)
        elif lazy_flag > 0:
            for i, zi in enumerate(z):
                if zi > 0:
                    self.caches[i].request(request)

        self.hits.append(hit)
        return hit

    def get_name(self):
        return "mLRU"


class LazyLRU(FemtoObj):
    def __init__(self, cache_size, catalog_size, caches_init, utilities, edges):
        super().__init__(cache_size, catalog_size, caches_init, LRU, utilities, edges)
        self.reset()

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

    def get_name(self):
        return "LazyLRU"


class BSA(FemtoObj):
    def __init__(self, cache_sizes, catalog_size, sample_size, utilities, edges, caches_init=None):
        # Init caches
        n_caches = edges.shape[1]

        if caches_init is None:
            caches_init = [np.random.random(catalog_size) for _ in range(n_caches)]

            for (i, cache_init), c_size in zip(enumerate(caches_init), cache_sizes):
                caches_init[i] = c_size * cache_init / np.sum(cache_init)
                # caches_init[i] = c_size * np.ones(catalog_size) / caches_init[i].size
        else:
            BSA_caches_init = []
            for cache_init in caches_init:
                BSA_cache_init = np.zeros(catalog_size)
                BSA_cache_init[cache_init] = 1
                BSA_caches_init.append(BSA_cache_init)
            caches_init = BSA_caches_init

        super().__init__(cache_sizes, catalog_size, caches_init, OGA, utilities, edges)
        self.sample_size = sample_size

        self.eta = np.sqrt(2*np.mean(cache_sizes) * self.num_cache / np.max(np.sum(edges, axis=0)) / sample_size ) / np.max(utilities)

        self.reset()


    def init_caches(self):
        self.caches = [self.cache_type(c_size, self.catalog_size, self.sample_size, cache_init, self.eta) for c_size, cache_init in zip(self.cache_size, self.caches_init)]


    def request(self, request, dest):
        hit = np.zeros(self.num_user_locs)
        local_edges = self.edges[dest, ]
        z = np.zeros(self.num_cache, dtype=np.float)

        request_cache_allocation = np.array([cache.get_cache()[request] for cache in self.caches])

        for idx in self.utilities_sorted_idx[dest]:
            if local_edges[idx] == 1: # Cache reachable
                z[idx] = min(request_cache_allocation[idx], 1-np.sum(z))

        hit[dest] = np.dot(z, local_edges*self.utilities[dest, ])

        # Find subgradient
        c = np.insert(request_cache_allocation, 0, 1) # Insert value 1 as allocation for the file for MBS
        A = np.concatenate((-np.ones((self.num_cache, 1)), -np.eye(self.num_cache)), axis=1)
        b = -self.utilities[dest,] * local_edges
        x = optimize.linprog(c=c, A_ub=A, b_ub=b, method="revised simplex").x # By default the bounds are (0, None), and thus the decision variables are nonnegative
        g = x[1:] # Exclude MBS
        # Update
        for subgradient, cache in zip(g, self.caches):
            if subgradient > 0.0001:
                cache.request(request, gradient=subgradient)


        self.hits.append(hit)
        return hit


    def get_name(self):
        return "BSA"


class DBSA(FemtoObj):
    def __init__(self, cache_sizes, catalog_size, sample_size, utilities, edges, caches_init=None):
        # Init caches
        n_caches = edges.shape[1]

        if caches_init is None:
            caches_init = [np.random.random(catalog_size) for _ in range(n_caches)]

            for (i, cache_init), c_size in zip(enumerate(caches_init), cache_sizes):
                caches_init[i] = c_size * cache_init / np.sum(cache_init)
                # caches_init[i] = c_size * np.ones(catalog_size) / caches_init[i].size
        else:
            BSA_caches_init = []
            for cache_init in caches_init:
                BSA_cache_init = np.zeros(catalog_size)
                BSA_cache_init[cache_init] = 1
                BSA_caches_init.append(BSA_cache_init)
            caches_init = BSA_caches_init

        super().__init__(cache_sizes, catalog_size, caches_init, DiscreteOGA, utilities, edges)
        self.sample_size = sample_size

        self.eta = np.sqrt(2*np.mean(cache_sizes) * self.num_cache / np.max(np.sum(edges, axis=0)) / sample_size ) / np.max(utilities)

        self.reset()


    def init_caches(self):
        self.caches = [self.cache_type(c_size, self.catalog_size, self.sample_size, cache_init, self.eta) for c_size, cache_init in zip(self.cache_size, self.caches_init)]


    def request(self, request, dest):
        hit = np.zeros(self.num_user_locs)
        local_edges = self.edges[dest, ]
        z = np.zeros(self.num_cache, dtype=np.float)

        request_cache_allocation = np.array([request in cache.get_cache() for cache in self.caches])

        for idx in self.utilities_sorted_idx[dest]:
            if local_edges[idx] == 1: # Cache reachable
                z[idx] = min(request_cache_allocation[idx], 1-np.sum(z))

        hit[dest] = np.dot(z, local_edges*self.utilities[dest, ])

        # Find subgradient
        # request_cache_allocation = np.array([cache.OGA.get_cache()[request] for cache in self.caches])
        c = np.insert(request_cache_allocation, 0, 1) # Insert value 1 as allocation for the file for MBS
        A = np.concatenate((-np.ones((self.num_cache, 1)), -np.eye(self.num_cache)), axis=1)
        b = -self.utilities[dest,] * local_edges
        x = optimize.linprog(c=c, A_ub=A, b_ub=b, method="revised simplex").x # By default the bounds are (0, None), and thus the decision variables are nonnegative
        g = x[1:] # Exclude MBS
        # Update
        for subgradient, cache in zip(g, self.caches):
            if subgradient > 0.0001:
                cache.request(request, gradient=subgradient)

        self.hits.append(hit)
        return hit


    def get_name(self):
        return "D-BSA"

class FemtoDEC(FemtoObj):
    def __init__(self, cache_sizes, catalog_size, caches_init, utilities, edges, expert_policies, mixing=True):
        super().__init__(cache_sizes, catalog_size, caches_init, None, utilities, edges)
        self.expert_policies = expert_policies
        self.mixing = mixing
        self.reset()


    def init_caches(self):
        self.caches = [RankingExperts(cache_size, self.catalog_size, cache_init, self.expert_policies, mixing=self.mixing) for cache_size, cache_init in zip(self.cache_size, self.caches_init)]
       

    def request(self, request, dest):
        hit = np.zeros(self.num_user_locs)
        local_edges = self.edges[dest, ]
        z = np.zeros(self.num_cache, dtype=np.int)
        lazy_flag = 0

        for idx in self.utilities_sorted_idx[dest]:
            if local_edges[idx] == 1: # Cache reachable
                if self.caches[idx].request(request) is True:                      
                    lazy_flag += 1


                    z[idx] = min(np.where(self.caches[idx].cache == request)[
                                0].size, 1-np.sum(z))
                    break

        hit[dest] = np.dot(z, local_edges*self.utilities[dest, ])


        # if lazy_flag == 0:
        #     for edge, cache in zip(local_edges, self.caches):
        #         if edge == 1: # Cache reachable
        #             cache.request(request)
        # elif lazy_flag > 0:
        #     updated = False
        #     for i, zi in enumerate(z):
        #         # Update the cache that is used for the request or the cache if a cache with a lower utility is used for the request
        #         if zi > 0 or (updated and local_edges[i]): 
        #             self.caches[i].request(request)
        #             updated = True

        self.hits.append(hit)
        return hit

    def get_name(self):
        return "FemtoDEC"

class FemtoDEC2(FemtoObj):
    def __init__(self, cache_sizes, catalog_size, caches_init, utilities, edges, expert_policies, mixing=True):
        super().__init__(cache_sizes, catalog_size, caches_init, None, utilities, edges)
        self.expert_policies = expert_policies
        self.mixing = mixing
        self.reset()


    def init_caches(self):
        self.caches = [RankingExperts(cache_size, self.catalog_size, cache_init, self.expert_policies, mixing=self.mixing) for cache_size, cache_init in zip(self.cache_size, self.caches_init)]
        

    def request(self, request, dest):
        hit = np.zeros(self.num_user_locs)
        local_edges = self.edges[dest, ]
        z = np.zeros(self.num_cache, dtype=np.float)

        request_cache_allocation = np.array([request in cache.get_cache() for cache in self.caches])

        for idx in self.utilities_sorted_idx[dest]:
            if local_edges[idx] == 1: # Cache reachable
                z[idx] = min(request_cache_allocation[idx], 1-np.sum(z))

        hit[dest] = np.dot(z, local_edges*self.utilities[dest, ])

        # Find subgradient
        c = np.insert(request_cache_allocation, 0, 1) # Insert value 1 as allocation for the file for MBS
        A = np.concatenate((-np.ones((self.num_cache, 1)), -np.eye(self.num_cache)), axis=1)
        b = -self.utilities[dest,] * local_edges
        x = optimize.linprog(c=c, A_ub=A, b_ub=b, method="revised simplex").x # By default the bounds are (0, None), and thus the decision variables are nonnegative
        g = x[1:] # Exclude MBS
        # Update
        for subgradient, cache in zip(g, self.caches):
            if subgradient > 0.0001:
                cache.request(request)

        self.hits.append(hit)
        return hit

    def get_name(self):
        return "FemtoDEC"



class femtoBH(FemtoObj):
    def __init__(self, cache_size, catalog_size, caches_init, utilities, edges, trace, destinations):
        # Note: This is a hardcoded approximation of the best-in-hindsight
        # Assumption utilities and edges are :
        # utilities = np.array([
        #     [1, 2, 100],
        #     [1, 2, 100],
        #     [1, 2, 100],
        #     [1, 2, 100]
        # ])
        # edges = np.array([
        #     [1, 1, 0],
        #     [1, 1, 0],
        #     [0, 1, 1],
        #     [0, 1, 1]
        # ]) # destinations x caches

        # Also only 3 caches
        # And

        caches_init = [[],[],[]]
        
        # Array of cache idx where sorted desceding on the cache with the highest cumulative utility
        caches_idx_sorted_utilities = np.argsort(np.sum(utilities, axis=0))[::-1] 
        for i, cache_idx in enumerate(caches_idx_sorted_utilities):
            object_count = np.zeros(catalog_size)
            for request, dest in zip(trace, destinations):
                if edges[dest, cache_idx] == 1:
                    # If the request is already in a connect cache with a higher utility don't count the request
                    for prev__cache_idx in caches_idx_sorted_utilities[:i]:
                        if edges[dest, prev__cache_idx] == 1 and request in caches_init[prev__cache_idx]:
                            break
                    else:
                        object_count[request] += 1
            caches_init[cache_idx] = np.argsort(object_count)[-cache_size[cache_idx]:]
        # print(caches_init)
        # exit(0)



        super().__init__(cache_size, catalog_size, caches_init, CacheStatic, utilities, edges)
        self.reset()

    def request(self, request, dest):
        hit = np.zeros(self.num_user_locs)  
        local_edges = self.edges[dest, ]
        z = np.zeros(self.num_cache, dtype=np.int)

        for idx in self.utilities_sorted_idx[dest]:
            if local_edges[idx] == 1: # Cache reachable
                if self.caches[idx].request(request):  
                    z[idx] = 1
                    break

        hit[dest] = np.dot(z, local_edges*self.utilities[dest, ])

        self.hits.append(hit)
        return hit

    def get_name(self):
        return "~BH"