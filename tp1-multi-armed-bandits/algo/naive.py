import numpy as np
from algo.utils import get_MAB, Regret_i

def compute_rewards(nb_groupe, groupe_size, MAB=get_MAB(5)):
    """
        Initialisation of the rewards for an Exploration phase
        following the number of groups and the size of each group
        Args:
            nb_groupe (int): number of groups
            groupe_size (int): size of each group 
    """
    rewards = np.zeros((nb_groupe, groupe_size))
    for groupe in range(nb_groupe):
        rewards[groupe] = [MAB[groupe].sample() for _ in range(groupe_size)]
    return rewards

def choose_bestgroupe(nb_groupe, groupe_size):
    """
        Initialisation of the rewards for an Exploration phase and
        choose the best group to continue the Exploitation phase
        following the number of groups and the size of each group

        Args:
            nb_groupe (int): number of groups
            groupe_size (int): size of each group
    """
    r = compute_rewards(nb_groupe, groupe_size)
    r_i = r.mean(axis=1)
    best_groupe = np.argmax(r_i)

    return best_groupe, r.flatten()

def compute_regret(l, vaccin, MAB, N=20, T=100):
    """
        Compute the regret for the phase of Exploration following
        the list of patient tested 
        Then, compute the regret for the phase of Exploitation
        Args:
            l (list): list of patient tested during the Exploration phase
            vaccin (int): index of the best group to continue the Exploitation phase
            MAB (list): list of the arms
    """
    list_vaccin = list(l.copy())
    mu_opti = np.max(list(map(lambda arm: arm.mean, MAB)))
    r = []

    ## Entrainement
    for i in range(1, N):
        r.append(Regret_i(i, mu_opti, list_vaccin[:i]))

    ## Exploration
    for i in range(T-N):
        list_vaccin.append(MAB[vaccin].sample())
        r.append(Regret_i(len(list_vaccin), mu_opti, list_vaccin[:N+i+1]))
    return r