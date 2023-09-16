import pytest
import numpy as np
from algo.naive import compute_rewards, choose_bestgroupe, compute_regret
from algo.utils import get_MAB

MAB = get_MAB(5)

def test_compute_rewards():
    rewards = compute_rewards(3, 4, MAB)
    assert rewards.shape == (3, 4)

def test_choose_bestgroupe():
    _, rewards = choose_bestgroupe(3, 4)
    assert len(rewards) == 12

def test_compute_regret():
    l = [arm.sample() for arm in MAB for _ in range(4)]
    vaccin = 2
    regret = compute_regret(l, vaccin, MAB)
    assert len(regret) == 99

def test_vaccin_list_length():
    l = [arm.sample() for arm in MAB for _ in range(4)]
    vaccin = 2
    compute_regret(l, vaccin, MAB)
    assert len(l) == 20

def test_exploitation_sampling():
    l = [arm.sample() for arm in MAB for _ in range(4)]
    best_groupe, _ = choose_bestgroupe(3, 4)
    for _ in range(10):
        l.append(MAB[best_groupe].sample())
    assert len(l) == 30

def test_best_groupe_choice():
    np.random.seed(0)
    nb_groupe, groupe_size = 3, 4
    rewards = compute_rewards(nb_groupe, groupe_size, MAB)
    best_groupe, _ = choose_bestgroupe(nb_groupe, groupe_size)
    assert best_groupe == np.argmax(rewards.mean(axis=1))
