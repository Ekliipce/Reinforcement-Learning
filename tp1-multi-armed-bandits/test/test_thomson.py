import pytest
import numpy as np

from algo.thomson import sample_thomson

class MockArm:
    def __init__(self, mean):
        self.mean = mean

    def sample(self):
        return np.random.binomial(1, self.mean)

def mock_regret_i(N, mu_opti, samples):
    return mu_opti - np.mean(samples)

def test_thomson_selection():
    MAB = [MockArm(0.2), MockArm(0.5), MockArm(0.7)]
    _, _ = sample_thomson(100, len(MAB), 0.7, MAB)

def test_beta_evolution():
    MAB = [MockArm(0.5), MockArm(0.5)]
    _, evolutions = sample_thomson(100, len(MAB), 0.5, MAB)
    assert not np.all(evolutions[0] == evolutions[-1])

def test_evolutions_length():
    MAB = [MockArm(0.2), MockArm(0.5), MockArm(0.7)]
    _, evolutions = sample_thomson(100, len(MAB), 0.7, MAB)
    assert len(evolutions) == 101

def test_positive_beta_values():
    MAB = [MockArm(0.2), MockArm(0.5), MockArm(0.7)]
    _, evolutions = sample_thomson(100, len(MAB), 0.7, MAB)
    assert np.all(evolutions >= 0)