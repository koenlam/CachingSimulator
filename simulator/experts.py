import random
import numpy as np

from .cache import CacheObj, LRU, LFU


class ExpertCache(CacheObj):
    def __init__(self, cache_size, catalog_size, cache_init):
        super().__init__(cache_size, catalog_size, cache_init)
        self.num_experts = 2
        self.experts = ("LRU", "LFU")
        self.expert_LRU = LRU(cache_size, catalog_size, cache_init)
        self.expert_LFU = LFU(cache_size, catalog_size, cache_init)
        self.request = self.request_WM
        self.eps = 0.01
        self.reset()

    def reset(self):
        super().reset()
        self.expert_choice = random.choice(self.experts)
        self.weights = dict(zip(self.experts, np.ones(self.num_experts)))
        self.expert_LRU.reset()
        self.expert_LFU.reset()
        self.expert_choices = []

    def get_expert_choices(self):
        choices = [self.experts.index(choice) for choice in self.expert_choices]
        return self.experts, choices

    def request_WM(self, request):
        # TODO: proper method to check if a file is in the cache without updating the cache
        hit_LRU = request in self.expert_LRU.get_cache()
        hit_LFU = request in self.expert_LFU.get_cache()

        is_hit = hit_LRU if self.expert_choice == "LRU" else hit_LFU

        # Save results
        self.hits.append(is_hit)
        self.expert_choices.append(self.expert_choice)

        # Adjust weights
        if hit_LRU is False:
            self.weights["LRU"] *= (1-self.eps)

        if hit_LFU is False:
            self.weights["LFU"] *= (1-self.eps)

        # Update the caches
        if self.weights["LRU"] >= self.weights["LFU"]:
            # Follow LRU
            self.expert_choice = "LRU"

            # Set the cache of LFU to be the same as LRU and update both experts using their respective policies
            # TODO: Does not work when the caches are stored in different ways (e.g. OGD and LRU)
            # Therefore an set_cache method has to be added to the cache objects
            self.expert_LFU.cache = self.expert_LRU.cache.copy()

            # Update the caches using their respective policies
            self.expert_LRU.request(request)
            self.expert_LFU.request(request)
        else:
            # Follow LFU
            self.expert_choice = "LFU"

            # Find difference between LRU and LFU caches and add those file into the LRU cache
            cache_diff = np.setdiff1d(
                self.expert_LFU.get_cache(), self.expert_LRU.get_cache())

            # There is a difference in caches
            for file_diff in cache_diff:
                self.expert_LRU.request(file_diff)

            # Update the caches using their respective policies
            self.expert_LRU.request(request)
            self.expert_LFU.request(request)
        return is_hit

    def get_name(self):
        return "ExpertCache"
