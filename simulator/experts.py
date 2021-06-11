import random
import numpy as np
import matplotlib.pyplot as plt

from .cache import CacheObj, LRU, LFU


class ExpertCache(CacheObj):
    def __init__(self, cache_size, catalog_size, cache_init, eps=0.01, alg="WM"):
        # Old version of the ExpertCache using resetting Virtual caches
        # Replaced with ExpertCacheEvict
        super().__init__(cache_size, catalog_size, cache_init)
        self.name = "ExpertCache " + str(alg)
        self.num_experts = 2
        self.experts = ("LRU", "LFU")
        self.expert_LRU = LRU(cache_size, catalog_size, cache_init)
        self.expert_LFU = LFU(cache_size, catalog_size, cache_init)
        self.eps = eps

        if alg == "WM":
            print("ExpertCache: Warning: WM is not correctly implemented")
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


    def plot_expert_choices(self):
        choices = [self.experts.index(choice) for choice in self.expert_choices]
        t = np.arange(1, len(choices)+1)
        plt.plot(t, choices, ".")
        plt.plot(t, np.array(self.weight_hist["LFU"]) / (np.array(self.weight_hist["LRU"]) + np.array(self.weight_hist["LFU"]) ))
        plt.plot(t, np.array(self.weight_hist["LRU"]) / (np.array(self.weight_hist["LRU"]) + np.array(self.weight_hist["LFU"]) ))
        plt.xlabel("Time")
        plt.ylabel("Choice")
        plt.legend(("Choice", "LFU weight (normalised)", "LRU weight (normalised)"))
        plt.yticks((0,1), self.experts)
        plt.show()


    def request(self, request):
        # TODO: proper method to check if a file is in the cache without updating the cache
        hit_LRU = request in self.expert_LRU.get_cache()
        hit_LFU = request in self.expert_LFU.get_cache()

        is_hit = request in self.cache


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


        # Reset "virtual caches"
        if self.expert_choice == "LRU":
            # Reset LFU cache to LRU

            # Set the cache of LFU to be the same as LRU and update both experts using their respective policies
            # TODO: Does not work when the caches are stored in different ways (e.g. OGD and LRU)
            # Therefore an set_cache method has to be added to the cache objects
            self.expert_LFU.cache = self.expert_LRU.cache.copy()

        elif self.expert_choice == "LFU":
            # Reset LRU to LFU

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
        else:
            raise ValueError("Expert choice not valid")


        # Update the "virtual"caches and actual caches
        self.expert_choice = self.choice_expert()

        self.expert_LRU.request(request)
        self.expert_LFU.request(request)

        self.cache = self.expert_LRU.get_cache() if self.expert_choice == "LRU" else self.expert_LFU.get_cache()

        return is_hit


class ExpertsCacheNeq(CacheObj):
    def __init__(self, cache_size, catalog_size, cache_init, experts, eps=0.01, alg="RWM", init_rnd_expert=True):
        super().__init__(cache_size, catalog_size, cache_init)
        self.name = "Expert Cache without equalization " + str(alg)
        self.num_experts = len(experts)
        self.expert_policies = experts
        self.eps = eps
        self.init_rnd_expert = init_rnd_expert
        self.alg = alg
            
        if alg in ("RWM", "Hedge", "Hedge_loss"):
            self.choice_expert = self.choice_expert_RWM
        else:
            raise ValueError(f"{alg} not implemented")

        self.reset()

    def reset(self):
        super().reset()
        self.experts = dict() # Experts salved as "expert name : expert"
        self.init_experts()
        
        self.expert_names = list(self.experts.keys())
        self.reset_weights()
        self.weights_hist = dict(zip(self.expert_names, ([] for _ in range(self.num_experts))))
        
        assert(self.init_rnd_expert in (True, False))
        if self.init_rnd_expert is True:
            self.cache = self.experts[random.choice(self.expert_names)]
        else:
            self.cache = self.experts[self.expert_names[0]]

        if self.alg == "Hedge_loss":
            self.loss = np.zeros(self.num_experts)
        
        
        self.expert_choices = []

    def reset_weights(self):
        self.weights = dict(zip(self.expert_names, np.ones(self.num_experts))) # Weights saved as "expert name : value"

    
    def init_experts(self):
        for policy in self.expert_policies:
            expert = policy(cache_size=self.cache_size, catalog_size=self.catalog_size, cache_init=self.cache_init)
            self.experts[expert.get_name()] = expert

    def choice_expert_RWM(self):
        weights = np.array(list(self.weights.values()))
        weight_dist = np.cumsum(weights/ np.sum(weights))
        chosen_expert_idx = np.where(random.random() <= weight_dist)[0][0]
        chosen_expert_name = self.expert_names[chosen_expert_idx]
        return self.experts[chosen_expert_name]


    def request(self, request, at_switch=True, weight_reset=False):
        # Check if hit
        is_hit = self.cache.get(request)

        # Save results
        self.hits.append(is_hit)
        self.expert_choices.append(self.cache.get_name())
        for expert in self.expert_names:
            self.weights_hist[expert].append(self.weights[expert])


        # Adjust weights and update caches
        if self.alg in ("RWM"):
            for expert in self.experts.values():
                self.weights[expert.get_name()] *= (1-self.eps*(1-float(expert.request(request))))
        elif self.alg == "Hedge":
            for expert in self.experts.values():
                loss = 1-float(expert.request(request))
                self.weights[expert.get_name()] *= np.exp(-self.eps*loss)
        elif self.alg == "Hedge_loss":
           for i, expert in enumerate(self.experts.values()):
                loss = 1-float(expert.request(request))
                self.loss[i] += loss


        # Choice the expert to follow for the next iteration
        if at_switch is True:
            if self.alg == "Hedge_loss":
                for i, expert in enumerate(self.experts.values()):
                    self.weights[expert.get_name()] *= np.exp(-self.eps*self.loss[i])
                self.loss = np.zeros(self.num_experts)
            self.cache = self.choice_expert()
            if weight_reset is True:
                self.reset_weights()
        elif at_switch is not False:
            raise ValueError(f"Switch value ({at_switch}) is not valid")

        return is_hit

class ExpertsCacheEvict(CacheObj):
    def __init__(self, cache_size, catalog_size, cache_init, eps=0.01, alg="WM", FTPL_eta=1.0):
        super().__init__(cache_size, catalog_size, cache_init)
        self.name = "Expert Cache using eviction experts " + str(alg)
        
        EvictFTPL_init = lambda catalog_size, cache_init: EvictFTPL(catalog_size, cache_init, FTPL_eta)

        self.expert_policies = (EvictLRU, EvictLFU, EvictFTPL_init)
        self.num_experts = len(self.expert_policies)
        self.eps = eps
        self.alg = alg

        if alg in ("WM"):
            self.choice_expert = self.choice_expert_WM
        elif alg in ("RWM", "Hedge"):
            self.choice_expert = self.choice_expert_RWM
        else:
            raise ValueError(f"Unknown algorithm {alg}")


        self.reset()

    def reset(self):
        super().reset()
        self.experts = [policy(self.catalog_size, self.cache_init) for policy in self.expert_policies]
        self.weights = np.ones(self.num_experts)
        self.weights_hist = [[] for _ in range(self.num_experts)]
        self.cache = self.cache_init.copy()
        self.prev_cache = self.cache_init.copy()
        self.expert_choice = random.randrange(0, self.num_experts)
        if self.alg == "WM":
            self.expert_choice = (self.expert_choice,) # WM requires tuples 
        self.expert_choices = []
        self.advice = [(None, None) for _ in range(self.num_experts)]

    def choice_expert_WM(self):
        advice_set = set(self.advice)
        advice_weights = []

        for advice in advice_set:
            weight = 0
            experts = []
            for i, expert_advice in enumerate(self.advice):
                if advice == expert_advice:
                    weight += self.weights[i]
                    experts.append(i)
            advice_weights.append((tuple(experts), weight))
        
        return max(advice_weights, key=lambda x: x[1])[0]
        

    def choice_expert_RWM(self):
        weight_dist = np.cumsum(self.weights / np.sum(self.weights))
        expert_idx = np.where(np.random.random() <= weight_dist)[0][0]
        return expert_idx

    def request(self, request):
        # Check for hit
        is_hit = request in self.cache

        # Adjust weights
        if self.alg in ("WM", "RWM"):
            for expert, (file2evict, file2add) in enumerate(self.advice):
                if not((request in self.prev_cache and request != file2evict) \
                    or (request not in self.prev_cache and request == file2add)):
                    self.weights[expert] *= (1-self.eps)
        elif self.alg == "Hedge":
            for expert, (file2evict, file2add) in enumerate(self.advice):
                if not((request in self.prev_cache and request != file2evict) \
                    or (request not in self.prev_cache and request == file2add)):
                    loss = 1 # Static loss of 1 for cache miss in hindsight
                    self.weights[expert] *= np.exp(-self.eps * loss)

        else:
            raise ValueError(f"Request: algorithm unknown")

        # Save results
        for expert, weight in enumerate(self.weights):
            self.weights_hist[expert].append(weight)
        self.hits.append(is_hit)
        self.expert_choices.append(self.expert_choice)

        # Choice expert
        self.expert_choice = self.choice_expert()
        self.advice = [expert.ask_advice(request, self.cache) for expert in self.experts]

        if isinstance(self.expert_choice, tuple):
            # Algorithm choses multiple experts with the same advice (WM)
            # Get the advice of the first one (doesn't matte which one is chosen)
             file2evict, file2add = self.advice[self.expert_choice[0]] 
        else:
            file2evict, file2add = self.advice[self.expert_choice]


        # Update cache
        self.prev_cache = self.cache.copy()

        self.cache = np.where(self.cache==file2evict, file2add, self.cache)

        if self.cache.dtype != np.int64:
            # print("Dtype changed")
            # TODO: Investigate why the dtype sometimes changes
            self.cache = self.cache.astype(np.int64)
            
        assert(self.cache.size == self.cache_size)
        assert(self.cache.dtype == np.int64)

        return is_hit
      
class EvictObj:
    def __init__(self, catalog_size, cache_init):
        self.name = "EvictObj"
        self.catalog_size = catalog_size
        self.cache_init = cache_init
        self.cache_size = len(cache_init)
        
    def get_name(self):
        return self.name


class EvictLRU(EvictObj):
    def __init__(self, catalog_size, cache_init):
        super().__init__(catalog_size, cache_init)
        self.name = "LRU"
        self.MAX_RECENCY = self.cache_size
        self.reset()

    def reset(self):
        # File_recency: the index corresponds to the file and the higher the value the more recent
        self.file_recency = np.zeros(self.catalog_size)
        
        for i, file in enumerate(self.cache_init):
            self.file_recency[file] = self.MAX_RECENCY - i

    def update(self, request):
        self.file_recency = np.where(self.file_recency < self.file_recency[request], self.file_recency, np.clip(self.file_recency-1, a_min=0, a_max=None))
        self.file_recency[request] = self.MAX_RECENCY
        assert(np.where(self.file_recency > 0)[0].size == self.cache_size)


    def ask_advice(self, request, cache):
        if request in cache:
            file2evict = None
            file2add = None
        else:
            # Look for the file to evict
            cache_recency = self.file_recency[cache]
            file2evict = cache[np.argmin(cache_recency)]
            file2add = request

        # Update the internal values
        self.update(request)
        return file2evict, file2add

class EvictLFU(EvictObj):
    def __init__(self, catalog_size, cache_init):
        super().__init__(catalog_size, cache_init)
        self.name = "LFU"
        self.reset()

    def reset(self):
        self.file_freq = np.zeros(self.catalog_size)

    def update(self, request):
        self.file_freq[request] += 1

    def ask_advice(self, request, cache):
        # Update the internal values
        self.update(request)

        file2evict = None
        file2add = None

        if request not in cache:
            # Look for the file to evict
            cache_file_freq = self.file_freq[cache]
            cache_file_freq_min = np.min(cache_file_freq)

            if self.file_freq[request] > cache_file_freq_min:
                # If the file request has a higher frequency than a file(s) in the cache
                # Evict the file and replace the request
                # If multiple file with the same low freq than choice a random file with low freq
                file2evict = cache[np.random.choice(np.where(cache_file_freq == cache_file_freq_min)[0])]
                file2add = request
        return file2evict, file2add

class EvictFTPL(EvictObj):
    def __init__(self, catalog_size, cache_init, eta=1.0):
        super().__init__(catalog_size, cache_init)
        self.name = "FTPL"
        self.eta = eta
        self.reset()

    def reset(self):
        self.file_freq = np.zeros(self.catalog_size)

    def update(self, request):
        self.file_freq[request] += 1

    def ask_advice(self, request, cache):
        #Update cache
        self.update(request)

        file2evict = None
        file2add = None

        if request not in cache:
            file_freq_perturbed = self.file_freq + self.eta*np.random.randn(self.file_freq.size)

            # Look for the file to evict
            cache_file_freq = file_freq_perturbed[cache]
            cache_file_freq_min = np.min(cache_file_freq)

            if file_freq_perturbed[request] > cache_file_freq_min:
                # If the file request has a higher frequency than a file(s) in the cache
                # Evict the file and replace the request
                # If multiple file with the same low freq than choice a random file with low freq
                file2evict = cache[np.random.choice(np.where(cache_file_freq == cache_file_freq_min)[0])]
                file2add = request
        return file2evict, file2add



        

