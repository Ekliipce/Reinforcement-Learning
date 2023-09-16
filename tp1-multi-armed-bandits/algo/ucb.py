import numpy as np
from algo.utils import get_MAB, Regret_i

def upper_bound(T, mu_opti, K, MAB, k=1):

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