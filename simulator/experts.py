import random
import numpy as np

from .cache import CacheObj, LRU, LFU


class ExpertCache(CacheObj):
    def __init__(self, cache_size, catalog_size, cache_init, eps= 0.01, alg="WM"):
        super().__init__(cache_size, catalog_size, cache_init)
        self.num_experts = 2
        self.experts = ("LRU", "LFU")
        self.expert_LRU = LRU(cache_size, catalog_size, cache_init)
        self.expert_LFU = LFU(cache_size, catalog_size, cache_init)
        self.eps = 0.01

        if alg == "WM":
            self.choice_expert = self.choice_expert_WM
        elif alg == "RWM":
            self.choice_expert = self.choice_expert_RWM
        else:
            raise ValueError("Chosen algorithm is invalid")

        self.reset()

    def reset(self):
        super().reset()
        self.expert_choice = random.choice(self.experts)
        self.weights = dict(zip(self.experts, np.ones(self.num_experts)))
        self.weight_hist = dict(zip(self.experts, ([] for _ in range(self.num_experts))))
        self.expert_LRU.reset()
        self.expert_LFU.reset()
        self.expert_choices = []

    def get_expert_choices(self):
        choices = [self.experts.index(choice) for choice in self.expert_choices]
        return self.experts, choices

    
    def choice_expert_WM(self):
        return "LRU" if self.weights["LRU"] >= self.weights["LFU"] else "LFU"
    
    def choice_expert_RWM(self):
        # Note: only work with 2 experts
        rnd = random.random()
        return "LRU" if rnd <= self.weights["LRU"] / sum(self.weights.values()) else "LFU"

    def request(self, request):
        # TODO: proper method to check if a file is in the cache without updating the cache
        hit_LRU = request in self.expert_LRU.get_cache()
        hit_LFU = request in self.expert_LFU.get_cache()

        is_hit = hit_LRU if self.expert_choice == "LRU" else hit_LFU

        # Save results
        self.hits.append(is_hit)
        self.expert_choices.append(self.expert_choice)
        for expert in self.experts:
            self.weight_hist[expert].append(self.weights[expert])

        # Adjust weights
        if hit_LRU is False:
            self.weights["LRU"] *= (1-self.eps)

        if hit_LFU is False:
            self.weights["LFU"] *= (1-self.eps)

        # Update the caches
        self.expert_choice = self.choice_expert()

        if self.expert_choice == "LRU":
            # Follow LRU

            # Set the cache of LFU to be the same as LRU and update both experts using their respective policies
            # TODO: Does not work when the caches are stored in different ways (e.g. OGD and LRU)
            # Therefore an set_cache method has to be added to the cache objects
            self.expert_LFU.cache = self.expert_LRU.cache.copy()

            # Update the caches using their respective policies
            self.expert_LRU.request(request)
            self.expert_LFU.request(request)
        else:
            # Follow LFU

            # Find difference between LRU and LFU caches and add those file into the LRU cache
            files_not_in_LRU = np.setdiff1d(
                self.expert_LFU.get_cache(), self.expert_LRU.get_cache())

            files_not_in_LFU = np.setdiff1d(
                self.expert_LRU.get_cache(), self.expert_LFU.get_cache())

            # Replace files that are not in LRU with the LFU ones
            # With this both caches have the same files
            for file_not_in_LFU, file_not_in_LRU in zip(files_not_in_LFU, files_not_in_LRU):
                swap_idx = np.where(self.expert_LRU.get_cache() == file_not_in_LFU)
                self.expert_LRU.cache[swap_idx] = file_not_in_LRU # TODO: Add proper replace method inside cache class

            # Verification for the caches to be equal
            # TODO: remove this for better performance
            # assert (np.sort(self.expert_LRU.get_cache()) == np.sort(self.expert_LFU.get_cache())).all()


            # # There is a difference in caches
            # for file_diff in cache_diff:
            #     self.expert_LRU.request(file_diff)

            # Update the caches using their respective policies
            self.expert_LRU.request(request)
            self.expert_LFU.request(request)
        return is_hit

    def get_name(self):
        return "ExpertCache"
