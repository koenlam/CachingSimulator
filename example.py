""" Example simulation of the caching simulator
"""



import matplotlib.pyplot as plt

from simulator import *


if __name__ == "__main__":
    relative_cache_size = 0.3
    catalog_size = 100
    cache_size = int(relative_cache_size*catalog_size)
    sample_size = 200000

    # Generate trace
    trace = gen_irm_trace(sample_size, catalog_size, power_law_exp=0.8)

    # Initialize caches
    cache_init = init_cache(cache_size, catalog_size)

    cache_LRU = LRU(cache_size, catalog_size, cache_init)
    cache_LFU = LFU(cache_size, catalog_size, cache_init)
    cache_BH = CacheStatic(cache_size, catalog_size, gen_best_static(trace, cache_size))
    cache_OGA = OGA(cache_size, catalog_size, sample_size, cache_init)
    cache_DiscreteOGA = DiscreteOGA(cache_size, catalog_size, sample_size, cache_init)
    cache_FTPL = FTPL(cache_size, catalog_size, cache_init, sample_size=sample_size)

    DiscreteOGA_init = lambda cache_size, catalog_size, cache_init: DiscreteOGA(cache_size, catalog_size, sample_size, cache_init)
    FTPL_init =  lambda cache_size, catalog_size, cache_init: FTPL(cache_size, catalog_size, cache_init, sample_size=sample_size)
    voting_expert_policies = (LRU, FTPL_init, DiscreteOGA_init)
    cache_voting_experts = VotingExperts(cache_size, catalog_size, cache_init, voting_expert_policies)

    RankOGA_init = lambda cache_size, catalog_size, cache_init : RankOGA(cache_size, catalog_size, cache_init, sample_size=sample_size)
    RankFTPL_init = lambda cache_size, catalog_size, cache_init : RankFTPL(cache_size, catalog_size, cache_init, sample_size=sample_size)
    ranking_expert_policies = (RankLRU, RankFTPL_init, RankOGA_init)
    cache_ranking_experts = RankingExperts(cache_size, catalog_size, cache_init, ranking_expert_policies,alg="SD", mixing=False)
    cache_ranking_experts_mix = RankingExperts(cache_size, catalog_size, cache_init, ranking_expert_policies,alg="SD", mixing=True)


    # Simulate caches
    caches = (cache_ranking_experts, cache_ranking_experts_mix, cache_voting_experts, cache_OGA, cache_DiscreteOGA, cache_FTPL, cache_LRU, cache_LFU, cache_BH)
    caches = simulate_caches(caches, trace)
    (cache_ranking_experts, cache_ranking_experts_mix, cache_voting_experts, cache_OGA, cache_DiscreteOGA, cache_FTPL, cache_LRU, cache_LFU, cache_BH) = caches

    plot_comp(
            cache_LRU,
            cache_LFU,
            cache_BH,
            cache_OGA,
            cache_DiscreteOGA,
            cache_FTPL,
            cache_voting_experts,
            cache_ranking_experts,
    )
    plt.show()
