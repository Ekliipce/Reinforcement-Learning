import numpy as np

from algo.utils import Regret_i
from scipy.stats import beta
from copy import deepcopy

def sample_thomson(T, K, mu_opti, MAB):
    """
        Thomson Sampling algorithm. 
        Use beta distribution to sample the arms and choose the best one
        at each time.
        Args:
            T: number of patients
            K: number of arms
            mu_opti: optimal mean
            MAB: list of arms
        Returns:
            list_regret: list of regret
            evolutions_beta: list of beta parameters of each arm at each time
    """
    list_patient =[[] for _ in range(K)]
    list_beta = [[1, 1] for _ in range(K)]
    list_regret = []
    evolutions_beta  = [deepcopy(list_beta)]

    for i in range(T):
        choice = np.argmax([np.random.beta(list_beta[k][0], list_beta[k][1]) for k in range(K)])
        result = MAB[choice].sample()

        if result:
            list_beta[choice][0] += 1
        else:
            list_beta[choice][1] += 1
        new_list_beta = deepcopy(list_beta)
        evolutions_beta.append(new_list_beta)
        

        list_patient[choice].append(result)
        list_regret.append(Regret_i(i, mu_opti, [x for sublist in list_patient for x in sublist]))

    return list_regret, np.array(evolutions_beta)