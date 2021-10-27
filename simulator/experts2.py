import random

import matplotlib.pyplot as plt
import numpy as np

from .cache import CacheObj


class VotingExperts(CacheObj):
    def __init__(self, cache_size, catalog_size, cache_init, expert_policies, eps=0.01):
        super().__init__(cache_size, catalog_size, cache_init)
        self.name = "VotingExperts"

        self.expert_policies = expert_policies

        self.num_experts = len(self.expert_policies)
        self.eps = eps

        self.reset()

    def reset(self):
        super().reset()
        self.experts  = [policy(self.cache_size, self.catalog_size, self.cache_init) for policy in self.expert_policies]
        self.weights = np.ones(self.num_experts) / self.num_experts

    def request(self, request):
        print(self.weights)
        is_hit = True if request in self.cache else False

        if not is_hit: # Cache miss            
            # Get the ranking of each file in the cache
            rankings = [expert.get_ranking() for expert in self.experts]
            cache_rankings = []
            for file in self.cache:
                file_ranking = 0
                for ranking, weight in zip(rankings, self.weights):
                    file_ranking += weight * ranking[file]
                cache_rankings.append(file_ranking)
            
            # Evict file with the lowest number of votes
            self.cache[np.argmin(cache_rankings)] = request

        # Update experts
        for i, expert in enumerate(self.experts):
            if not expert.request(request): # Cache miss for the expert
                self.weights[i] *= (1-self.eps)

        # Normalize weights
        total_weights = np.sum(self.weights)
        for i, weight in enumerate(self.weights):
            self.weights[i] = weight / total_weights

        self.update_perf_metrics(is_hit)
        return is_hit
