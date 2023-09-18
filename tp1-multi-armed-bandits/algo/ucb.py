import numpy as np
from algo.utils import get_MAB, Regret_i

def upper_bound(T, mu_opti, K, MAB, k=1):
    """
        UCB algorithm.
        Use upper bound to choose the best arm at each time.
        Args:
            T: number of patients
            mu_opti: optimal mean
            K: number of arms
            MAB: list of arms
            k: hyperparameter of the upper bound
        Returns:
            list_regret: list of regret
            list_choice: list of vaccin choisen at each time
    """
    list_patient =[[] for _ in range(K)]
    list_a = [10 for _ in range(K)]
    list_regret = []
    list_choice = []
    
    for i in range(1, T):
        a = np.argmax(list_a)
        
        list_patient[a].append(MAB[a].sample())
        
        N = i
        T_i = len(list_patient[a])
        X_i = np.mean(list_patient[a])
        biais = np.sqrt(2 * np.log(N) / T_i)
        list_a[a] = X_i + k * biais
        
        list_regret.append(Regret_i(N, mu_opti, [x for sublist in list_patient for x in sublist]))
        list_choice.append(a)

    return list_regret, list_choice