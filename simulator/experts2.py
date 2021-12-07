import random

import matplotlib.pyplot as plt
import numpy as np


from .cache import CacheObj, OGA


class VotingExperts(CacheObj):
    def __init__(self, cache_size, catalog_size, cache_init, expert_policies, eps=1/np.e, normalize=True, mixing=False):
        super().__init__(cache_size, catalog_size, cache_init)
        self.name = "ACME"

        self.expert_policies = expert_policies

        self.num_experts = len(self.expert_policies)
        self.eps = eps # Expert update parameter
        self.alpha = 0.005 # Mixing parameter
        self.normalize=normalize
        self.mixing = mixing

        self.reset()

    def reset(self):
        super().reset()
        self.experts  = [policy(self.cache_size, self.catalog_size, self.cache_init) for policy in self.expert_policies]
        self.weights = np.ones(self.num_experts) / self.num_experts
        self.past_avg_weights = np.zeros(self.num_experts)
        self.past_weights = [self.weights.copy()]


    def plot_expert_weights(self):
        t = np.arange(1, len(self.past_weights)+1)

        past_weights = np.array(self.past_weights)

        expert_names = []
        for i, expert in enumerate(self.experts):
            expert_names.append(expert.get_name())
            plt.plot(t, past_weights[:,i], '.')
        plt.legend(expert_names)
        plt.xlabel("Time")
        plt.ylabel("Weights")
        

    def request(self, request):
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
        if self.normalize is True:
            total_weights = np.sum(self.weights)
            for i, weight in enumerate(self.weights):
                self.weights[i] = weight / total_weights

        # Mixing update
        if self.mixing is True:
            self.weights_n = (1-self.alpha)*self.weights + self.alpha*self.past_avg_weights
            self.past_avg_weights = (len(self.hits)*self.past_avg_weights + self.weights) / (len(self.hits)+1)
            self.weights = self.weights_n
        
        # Update perf metrics
        self.update_perf_metrics(is_hit)
        self.past_weights.append(self.weights)
        return is_hit



class RankingExperts(CacheObj):
    def __init__(self, cache_size, catalog_size, cache_init, expert_policies, alg="SD", eps=1/np.e, rank_threshold=None, normalize=True, mixing=True):
        super().__init__(cache_size, catalog_size, cache_init)
        self.name = f"DEC ({alg})"

        self.expert_policies = expert_policies
        self.num_experts = len(self.expert_policies)
        

        # How high has a expert to rank an object for it to be correct
        self.rank_threshold = rank_threshold if rank_threshold is not None else cache_size
        self.alg = alg
        self.eps = eps # Expert update parameter
        self.alpha = 0.005 # Mixing parameter
        self.normalize = normalize
        self.mixing = mixing

        self.reset()


    def reset(self):
        super().reset()
        self.experts  = [policy(self.cache_size, self.catalog_size, self.cache_init) for policy in self.expert_policies]
        self.weights = np.ones(self.num_experts) / self.num_experts
        self.past_avg_weights = np.zeros(self.num_experts)
        self.past_weights = []
        self.current_expert_id = 0 # Since all experts are the initial cache it doesn't really matter which one is currently chosen
        self.expert_choices = []
        self.caches=[]

        if self.alg == "FTL":
            self.FTPL_perturbation = 0
            self.choose_expert = self.choose_expert_FTL
        elif self.alg == "FTPL":
            self.FTPL_perturbation = 0.1
            self.choose_expert = self.choose_expert_FTL
        elif self.alg == "RWM":
            self.choose_expert = self.choose_expert_RWM
        elif self.alg == "SD":
            self.choose_expert =self.choose_expert_SD
        elif self.alg == "SD-Hedge":
            self.choose_expert =self.choose_expert_SD
            self.past_avg_loss = np.zeros(self.num_experts)
        else:
            raise ValueError("{self.alg} not valid. Valid algorithms: ('WM', 'RWM', 'SD')")

    
    def plot_expert_weights(self):
        t = np.arange(1, len(self.past_weights)+1)

        past_weights = np.array(self.past_weights)

        expert_names = []
        for i, expert in enumerate(self.experts):
            expert_names.append(expert.get_name())
            plt.plot(t, past_weights[:,i], '.')
        plt.legend(expert_names)
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0), useMathText=True)
        plt.xlabel("Time")
        plt.ylabel("Weights")


    def choose_expert_FTL(self):
        return np.argmax(self.weights + self.FTPL_perturbation*np.random.randn(self.weights.size))


    def choose_expert_RWM(self):
        weight_dist = np.cumsum(self.weights/ np.sum(self.weights))
        chosen_expert_idx = np.where(random.random() <= weight_dist)[0][0]
        return chosen_expert_idx


    def choose_expert_SD(self):
        # Check if there is an past weight. Otherwise t
        if len(self.past_weights) > 1:
            past_weights = self.past_weights[-1]
        else:
            return self.choose_expert_RWM()
        
        # With probability (current weight / past_weight) do not change expert 
        if random.random() <= self.weights[self.current_expert_id] / past_weights[self.current_expert_id]:
            return self.current_expert_id
        else:
            return self.choose_expert_RWM()

    def request(self, request):
        is_hit = True if request in self.cache else False


        # [expert.update(request) for expert in self.experts]

        # rankings = [expert.get_ranking() for expert in self.experts]


        # Update expert ranking and weights
        for i, expert in enumerate(self.experts):
            # request_importance = rankings[i][request]
            # # print(f"{expert.get_name():} | {request} | {request_importance}")
            # # request_importance = request_importance if request_importance > (self.catalog_size-self.cache_size) else 0
            # request_importance = request_importance if request_importance > 0.6 else 0
            # # request_importance = 1.0 if request_importance > 0.6 else 0

            # loss = 1-(request_importance)
            # self.weights[i] *= np.exp(-self.eps*loss)

            expert_correct = True if request in expert.get_sorted_ranking()[-self.rank_threshold:] else False
            
            if self.alg == "SD-Hedge":
                loss = (1-self.alpha)*(not expert_correct) + self.alpha*self.past_avg_loss[i]
                self.past_avg_loss[i] = (len(self.hits)*self.past_avg_loss[i] + (not expert_correct)) / (len(self.hits)+1)
                self.weights[i] *= np.exp(-self.eps*loss)

            else:
                if not expert_correct:
                    self.weights[i] *= (1-self.eps)
            expert.update(request)

        # Normalize weights
        if self.normalize is True:
            total_weights = np.sum(self.weights)
            for i, weight in enumerate(self.weights):
                self.weights[i] = weight / total_weights

        # Mixing update
        if self.mixing is True:
            self.weights_n = (1-self.alpha)*self.weights + self.alpha*self.past_avg_weights
            self.past_avg_weights = (len(self.hits)*self.past_avg_weights + self.weights) / (len(self.hits)+1)
            self.weights = self.weights_n


        if not is_hit: # Cache miss
            # Choose expert
            self.current_expert_id = self.choose_expert()

            # ranking = rankings[expert_id]

            ranking = self.experts[self.current_expert_id].get_ranking()
            cache_ranking = ranking[self.cache]

            # Evict lowest rank if ranking is lower than the ranking of request
            if ranking[request] >= np.min(cache_ranking):
                self.cache[np.argmin(cache_ranking)] = request

          
        # Update perf metrics
        self.past_weights.append(self.weights.copy())
        self.expert_choices.append(self.current_expert_id)
        self.update_perf_metrics(is_hit)
        # self.caches.append(self.cache.copy())
        return is_hit

        



class RankObj:
    def __init__(self, cache_size, catalog_size, cache_init):
        self.name = "RankObj"
        self.cache_size = cache_size
        self.catalog_size = catalog_size
        self.cache_init = cache_init

    def get_name(self):
        return self.name


class RankStatic(RankObj):
    def __init__(self, cache_size, catalog_size, cache_init):
        super().__init__(cache_size, catalog_size, cache_init)
        self.name = "CacheStatic"
        self.ranking = None

    def get_ranking_old(self):
        if self.ranking is None:
            ranking = np.zeros(self.catalog_size)
            cache_ranks = np.arange(self.catalog_size-self.cache_size+1, self.catalog_size+1) 
            assert cache_ranks.size == self.cache_size
            for ranks, c in zip(cache_ranks, self.cache_init):
                ranking[c] =  ranks/self.catalog_size
            self.ranking = ranking
        return self.ranking
    
    def get_ranking(self):
        ranking = np.zeros(self.catalog_size)
        ranking[self.cache_init] = 1
        return ranking

    def get_sorted_ranking(self):
        files = np.arange(self.catalog_size)
        # Return sorted array of files where the higher the idx the higher the ranking of the file
        return np.concatenate((np.delete(files, self.cache_init), self.cache_init)) 

    def update(self, request):
        # Static cache doesn't update
        pass


class RankLRU(RankObj):
    def __init__(self, cache_size, catalog_size, cache_init):
        super().__init__(cache_size, catalog_size, cache_init)
        self.name = "LRU"
        self.MAX_RECENCY = self.catalog_size
        self.reset()

    def reset(self):
        # File_recency: the index corresponds to the file and the higher the value the more recent
        self.file_recency = np.zeros(self.catalog_size)
        
        for i, file in enumerate(self.cache_init):
            self.file_recency[file] = self.cache_size - i

    def get_ranking_old(self):
        ranking = self.file_recency

        # assert np.sum(ranking) ==  np.sum(np.arange(1, self.cache_size+1))
        # Scale ranking to be between 0 and 1
        ranking /= self.catalog_size
        return ranking

    def get_ranking(self):
        return self.file_recency
    

    def get_sorted_ranking(self):
        return np.argsort(self.file_recency)


    def update(self, request):
        self.file_recency = np.where(self.file_recency < self.file_recency[request], self.file_recency, np.clip(self.file_recency-1, a_min=0, a_max=None))
        self.file_recency[request] = self.MAX_RECENCY


class RankLFU(RankObj):
    def __init__(self, cache_size, catalog_size, cache_init):
        super().__init__(cache_size, catalog_size, cache_init)
        self.name = "LFU"
        self.reset()

    def reset(self):
        self.file_freq = np.zeros(self.catalog_size)

    def get_ranking_old(self):
        # return np.arange(1, self.catalog_size+1)[np.argsort(self.file_freq)]
        # ranking_raw = self.file_freq
        # ranking_raw_argsort = np.argsort(ranking_raw)

        # ranking = np.zeros(self.catalog_size)
        # for rank, idx in zip(np.arange(1, self.catalog_size+1), ranking_raw_argsort):
        #     ranking[idx] = rank
        # ranking = rankdata(self.file_freq)

        sort = np.argsort(self.file_freq)
        ranking = np.empty(sort.size)
        ranking[sort] = np.arange(1, self.catalog_size+1)

        # assert np.sum(ranking) ==  np.sum(np.arange(1, self.catalog_size+1))
        # Scale ranking to be between 0 and 1
        ranking /= self.catalog_size
        return ranking

    def get_ranking(self):
        return self.file_freq

    def get_sorted_ranking(self):
        return np.argsort(self.file_freq)

    def update(self, request):
        self.file_freq[request] += 1


class RankFTPL(RankObj):
    def __init__(self, cache_size, catalog_size, cache_init, eta=None, sample_size=None):
        super().__init__(cache_size, catalog_size, cache_init)
        self.name = "FTPL"
        
        if eta is None:
            if sample_size is not None:
                self.eta = np.sqrt(sample_size/cache_size)/(4*np.pi*np.log(catalog_size))
            else:
                self.eta = 1.0
        else:
            self.eta = eta
        self.reset()

    def reset(self):
        self.file_freq = np.zeros(self.catalog_size)


    def get_ranking(self):
        return self.file_freq + self.eta*np.random.randn(self.file_freq.size)

    def get_sorted_ranking(self):
        return np.argsort(self.file_freq + self.eta*np.random.randn(self.file_freq.size))

    def update(self, request):
        self.file_freq[request] += 1

class RankOGA(RankObj):
    def __init__(self, cache_size, catalog_size, cache_init, sample_size=None):
        super().__init__(cache_size, catalog_size, cache_init)
        self.name = "Discrete OGA"
        self.sample_size = sample_size
        self.reset()


    def get_ranking_old(self):
        """ Ranking: value of catalog_size for the most important and 1 for the least important"""
        # return np.arange(1, self.catalog_size+1)[np.argsort(self.OGA.cache)]
        # ranking_raw = self.OGA.cache
        # ranking_raw_argsort = np.argsort(ranking_raw)

        # ranking = np.zeros(self.catalog_size)
        # for rank, idx in zip(np.arange(1, self.catalog_size+1), ranking_raw_argsort):
        #     ranking[idx] = rank
        

        sort = np.argsort(self.OGA.cache)
        ranking = np.empty(sort.size)
        ranking[sort] = np.arange(1, self.catalog_size+1)

        # assert np.sum(ranking) ==  np.sum(np.arange(1, self.catalog_size+1))
        # Scale ranking to be between 0 and 1
        ranking /= self.catalog_size
        return ranking


    def get_ranking(self):
        return self.OGA.cache

    def get_sorted_ranking(self):
        return np.argsort(self.OGA.cache)


    def reset(self):
        self.OGA = OGA(cache_size=self.cache_size, catalog_size=self.catalog_size, cache_init=self.cache_init, sample_size=self.sample_size)
    
    def update(self, request):
        self.OGA.request(request)