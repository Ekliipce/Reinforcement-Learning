import pytest
import numpy as np
from algo.ucb import upper_bound

class MockArm:
    def __init__(self, mean):
        self.mean = mean

    def sample(self):
        return self.mean

def test_ucb_selection():
    MAB = [MockArm(0.2), MockArm(0.5), MockArm(0.7)]
    _, choices = upper_bound(100, 0.7, len(MAB), MAB)
    optimal_arm_chosen = sum([1 for choice in choices if choice == 2])
    assert optimal_arm_chosen > len(choices) * 0.5 

def test_ucb_value():
    MAB = [MockArm(0.2), MockArm(0.5), MockArm(0.7)]
    T = 100
    _, choices = upper_bound(T, 0.7, len(MAB), MAB)

    T_2 = sum([1 for choice in choices if choice == 2])
    biais = np.sqrt(2 * np.log(T) / T_2)
    assert 0 < biais < 1  

def test_convergence_to_optimal_arm():
    MAB = [MockArm(0.2), MockArm(0.5), MockArm(0.7)]
    _, choices = upper_bound(1000, 0.7, len(MAB), MAB)
    assert choices[-1] == 2 

