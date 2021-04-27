import random
import numpy as np
import matplotlib.pyplot as plt

from .cache import CacheObj, LRU, LFU


class ExpertCache(CacheObj):
    def __init__(self, cache_size, catalog_size, cache_init, eps=0.01, alg="WM"):
        super().__init__(cache_size, catalog_size, cache_init)
        self.name = "ExpertCache " + str(alg)
        self.num_experts = 2
        self.experts = ("LRU", "LFU")
        self.expert_LRU = LRU(cache_size, catalog_size, cache_init)
        self.expert_LFU = LFU(cache_size, catalog_size, cache_init)
        self.eps = eps

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
    def __init__(self, cache_size, catalog_size, cache_init, experts, eps=0.01, alg="WM"):
        super().__init__(cache_size, catalog_size, cache_init)
        self.name = "Expert Cache without equalization " + str(alg)
        self.num_experts = len(experts)
        self.expert_policies = experts
        self.eps = eps

        self.reset()

    def reset(self):
        super().reset()
        self.experts = dict()
        self.init_experts()
        
        self.expert_names = list(self.experts.keys())
        self.weights = dict(zip(self.expert_names, np.ones(self.num_experts)))
        self.weights_hist = dict(zip(self.expert_names, ([] for _ in range(self.num_experts))))
        
        self.cache = self.experts[random.choice(self.expert_names)]
        self.expert_choices = []

    
    def init_experts(self):
        for policy in self.expert_policies:
            expert = policy(cache_size=self.cache_size, catalog_size=self.catalog_size, cache_init=self.cache_init)
            self.experts[expert.get_name()] = expert

    def choice_expert(self):
        return self.experts[max(self.weights.items(), key=lambda x: x[1])[0]]

    def request(self, request):
        # Check if hit
        is_hit = self.cache.get(request)

        # Save results
        self.hits.append(is_hit)
        self.expert_choices.append(self.cache.get_name())
        for expert in self.expert_names:
            self.weights_hist[expert].append(self.weights[expert])


        # Adjust weights and update caches
        for expert in self.experts.values():
            self.weights[expert.get_name()] *= (1-self.eps*(1-float(expert.request(request))))

        # Choice the expert to follow for the next iteration
        self.cache = self.choice_expert()

        return is_hit

class ExpertsCacheEvict(CacheObj):
    def __init__(self, cache_size, catalog_size, cache_init, eps=0.01, alg="WM"):
        super().__init__(cache_size, catalog_size, cache_init)
        self.name = "Expert Cache using eviction experts " + str(alg)
        self.expert_policies = (EvictLRU, EvictLFU, EvictFTPL)
        self.num_experts = len(self.expert_policies)
        self.eps = eps

        if alg == "WM":
            self.alg = "WM"
            self.choice_expert = self.choice_expert_WM
        elif alg in ("WM-HS", "RWM-HS", "RWM"):
            # Weighted majority Hindsight
            raise ValueError(f"{alg} not implemented")
        else:
            raise ValueError(f"Unknown algorithm {alg}")


        self.reset()

    def reset(self):
        super().reset()
        self.experts = [policy(self.catalog_size, self.cache_init) for policy in self.expert_policies]
        self.weights = np.ones(self.num_experts)
        self.weights_hist = [[] for _ in range(self.num_experts)]
        self.cache = self.cache_init
        self.expert_choice = random.randrange(0, self.num_experts)
        self.expert_choices = []

    def choice_expert_WM(self):
        return max(enumerate(self.weights), key=lambda x: x[1])[0]

    def request(self, request):
        # Check for hit
        is_hit = request in self.cache

        # Adjust weights
        if not is_hit:
            self.weights[self.expert_choice] *= (1-self.eps)

        # Save results
        for expert, weight in enumerate(self.weights):
            self.weights_hist[expert].append(weight)
        self.hits.append(is_hit)
        self.expert_choices.append(self.expert_choice)

        # Choice expert
        self.expert_choice = self.choice_expert()
        files2evict, files2add = self.experts[self.expert_choice].ask_advice(request, self.cache)

        if type(files2evict) != np.ndarray:
            self.cache = np.where(self.cache==files2evict, files2add, self.cache)
        else:
            # Update cache
            for file2evict, file2add in zip(files2evict, files2add):
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
            self.file_recency[file] += i + 1 # Due to the index starting at zero, one it added to differentiate it from the not in the cache

    def update(self, request):
        self.file_recency = np.clip(self.file_recency-1, a_min=0, a_max=None)
        self.file_recency[request] = self.MAX_RECENCY


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
                file2evict = np.random.choice(np.where(cache_file_freq == cache_file_freq_min)[0])
                file2add = request
        return file2evict, file2add

class EvictFTPL(EvictObj):
    def __init__(self, catalog_size, cache_init):
        super().__init__(catalog_size, cache_init)
        self.name = "FTPL"
        self.reset()

    def reset(self):
        self.file_freq = np.zeros(self.catalog_size)

    def update(self, request):
        self.file_freq[request] += 1

    def ask_advice(self, request, cache):
        #Update cache
        self.update(request)

        y = self.file_freq + np.random.randn(self.file_freq.size)
        new_cache = np.argsort(y)[-self.cache_size:]

        file2evict = np.setdiff1d(cache, new_cache)
        file2add = np.setdiff1d(new_cache, cache)

        file2evict = file2evict if file2evict.size else None
        file2add = file2add if file2add.size else None

        return file2evict, file2add



        

