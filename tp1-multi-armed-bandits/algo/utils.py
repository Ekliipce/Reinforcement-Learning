import numpy as np
import matplotlib.pyplot as plt

class ArmBernoulli:
    def __init__(self, p: float, random_state: int = 0):
        """
        Bernoulli arm
        Args:
             p (float): mean parameter
             random_state (int): seed to make experiments reproducible
        """
        self.mean = p
        self.local_random = np.random.RandomState(random_state)
        
    def sample(self, printable = False):
        random = self.local_random.rand()
        
        if printable:
            print (f"random: {random}, mean: {self.mean}")
        return random < self.mean

def get_MAB(K):
    """
    Generate a list of K arms with random means
    Args:
        K (int): number of arms
    Returns:
        MAB (list): list of K arms
    """
    means = np.random.random(K)
    MAB = [ArmBernoulli(mean) for mean in means]
    return MAB

def Regret_i(N, mu, list_vaccine):
    """
        Compute the regret at step i
            N : number of patients tested
            mu : mean reward of the best vaccin
            list_vaccine : list of rewards for each patient
    """
    E = np.sum(list_vaccine) 
    return N * mu - E