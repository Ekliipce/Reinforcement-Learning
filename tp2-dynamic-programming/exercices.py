"""
Ce fichier contient des exercices à compléter sur la programmation dynamique.
Il est évalué automatiquement avec pytest, vous pouvez le lancer avec la
commande `pytest exercices.py`.
"""
import typing as t
import random
import pytest
import numpy as np
import gym
from gym import spaces
from copy import deepcopy as dp



"""
Partie 1 - Processus décisionnels de Markov
===========================================

Rappel: un processus décisionnel de Markov (MDP) est un modèle de
décision séquentielle dans lequel les états du système sont décrits par
une variable aléatoire, et les actions du système sont décrites par une
autre variable aléatoire. Les transitions entre états sont décrites par
une matrice de transition, et les récompenses associées à chaque
transition sont décrites par une matrice de récompenses.

Dans ce TP, nous allons utiliser la librairie `gym` pour implémenter un
MDP simple, et nous allons utiliser la programmation dynamique pour
résoudre ce MDP.
"""

# Exercice 1: MDP simple
# ----------------------
# Implémenter un MDP simple avec la librairie `gym`. Ce MDP doit avoir
# 3 états, 2 actions, et les transitions et récompenses suivantes:
#   - état 0, action 0 -> état 1, récompense -1
#   - état 0, action 1 -> état 0, récompense -1
#   - état 1, action 0 -> état 0, récompense -1
#   - état 1, action 1 -> état 2, récompense -1
#   - état 2, action 0 -> état 2, récompense 0
#   - état 2, action 1 -> état 0, récompense -1


class MDP(gym.Env):
    """
    MDP simple avec 3 états et 2 actions.
    """

    observation_space: spaces.Discrete
    action_space: spaces.Discrete

    # state, action -> [(next_state, reward, done)]
    P: list[list[tuple[int, float, bool]]]

    def __init__(self):
        self.observation_space = spaces.Discrete(3)
        self.action_space = spaces.Discrete(2)
        self.state = random.randint(0, 2)

        self.P  = [[(1, -1, False), (0, -1, False)], 
                   [(0, -1, False), (2, -1, False)],
                   [(2, 0, False), (0, -1, False)]]

    def step(self, action: int) -> tuple[int, float, bool, dict]:  # type: ignore
        """
        Effectue une transition dans le MDP.
        Renvoie l'observation suivante, la récompense, un booléen indiquant
        si l'épisode est terminé, et un dictionnaire d'informations.
        """
        transition = self.P[self.state][action]
        self.state = transition[0]

        return (transition[0], transition[1], transition[2], {"info":"It is an info"})

    def reset(self):
        """
        Réinitialise l'environnement à un état initial
        """
        self.state = 0
        return self.state        


# Tests pour l'exercice 1
def test_mdp():
    mdp = MDP()
    assert mdp.P[0][0] == (1, -1, False)
    assert mdp.P[0][1] == (0, -1, False)
    assert mdp.P[1][0] == (0, -1, False)
    assert mdp.P[1][1] == (2, -1, False)
    assert mdp.P[2][0] == (2, 0, False)
    assert mdp.P[2][1] == (0, -1, False)

    mdp.reset()
    ret = mdp.step(0)
    assert ret[0] in [0, 1, 2]
    assert ret[1] in [0, -1]
    assert ret[2] in [True, False]
    assert isinstance(ret[3], dict)


# Exercice 2: Résolution du MDP
# -----------------------------
# Ecrire une fonction qui calcule la valeur de chaque état du MDP, en
# utilisant la programmation dynamique.
# L'algorithme de programmation dynamique est le suivant:
#   - Initialiser la valeur de chaque état à 0
#   - Tant que la valeur de chaque état n'a pas convergé:
#       - Pour chaque état:
#           - Estimer la fonction de valeur de chaque état
#           - Choisir l'action qui maximise la valeur
#           - Mettre à jour la valeur de l'état
#
# Indice: la fonction doit être itérative.

def compute_value(array, state, action, mdp: MDP, gamma: float):
    reward = mdp.P[state][action][1]
    next_state = mdp.P[state][action][0]
    return reward + gamma * array[next_state]

def mdp_value_iteration(mdp: MDP, max_iter: int = 1000, gamma=1.0) -> np.ndarray:
    """
    Estimation de la fonction de valeur grâce à l'algorithme "value iteration":
    https://en.wikipedia.org/wiki/Markov_decision_process#Value_iteration
    """
    values = np.zeros(mdp.observation_space.n)
    #print(f"============> Values: {values}\n")
    for i in range(max_iter):
        change = 0
        
        for state in range(mdp.observation_space.n):
            choices = [compute_value(values, state, action, mdp, gamma) 
                       for action in range(mdp.action_space.n)]
            new_value = max(choices)
            
            #print (f"============> New value: {new_value}\n")
            
            if (new_value != values[state]):
                values[state] = new_value
                change += 1

        if (change == 0 or i >= max_iter):
            break
    return values


def test_mdp_value_iteration():
    mdp = MDP()
    values = mdp_value_iteration(mdp, max_iter=1000, gamma=1.0)
    assert np.allclose(values, [-2, -1, 0])
    values = mdp_value_iteration(mdp, max_iter=1000, gamma=0.9)
    assert np.allclose(values, [-1.9, -1, 0])


# Exercice 3: Extension du MDP à un GridWorld (sans bruit)
# --------------------------------------------------------
# Implémenter un MDP simple avec la librairie `gym`. Ce MDP est formé
# d'un GridWorld de 3x4 cases, avec 4 actions possibles (haut, bas, gauche,
# droite). La case (1, 1) est inaccessible (mur), tandis que la case (1, 3)
# est un état terminal avec une récompense de -1. La case (0, 3) est un état
# terminal avec une récompense de +1. Tout autre état a une récompense de 0.
# L'agent commence dans la case (0, 0).

# Complétez la classe ci-dessous pour implémenter ce MDP.
# Puis, utilisez l'algorithme de value iteration pour calculer la fonction de
# valeur de chaque état.

class GridWorldEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    # F: Free, S: Start, P: Positive reward, N: negative reward, W: Wall
    grid: np.ndarray = np.array(
        [
            ["F", "F", "F", "P"],
            ["F", "W", "F", "N"],
            ["F", "F", "F", "F"],
            ["S", "F", "F", "F"],
        ]
    )
    current_position: tuple[int, int] = (0, 0)

    def __init__(self):
        super(GridWorldEnv, self).__init__()

        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = spaces.Tuple((spaces.Discrete(4), spaces.Discrete(4)))

        self.current_position = (0, 0)

    def step(self, action, movable=False):
        # if self.grid[tuple(self.current_position)] == "N":
        #     return None, -1, True
        # elif self.grid[tuple(self.current_position)] == "P":
        #     return None, 1, True 


        if action == 0:  # Up
            position = (
                max(0, self.current_position[0] - 1),
                self.current_position[1],
            )
        elif action == 1:  # Down
            position = (
                min(3, self.current_position[0] + 1),
                self.current_position[1],
            )
        elif action == 2:  # Left
            position = (
                self.current_position[0],
                max(0, self.current_position[1] - 1),
            )
        elif action == 3:  # Right
            position = (
                self.current_position[0],
                min(3, self.current_position[1] + 1),
            )

        next_state = tuple(position)

        # Check if the agent has reached the goal
        is_done = self.grid[tuple(position)] in {"P", "N"}

        # Provide reward
        if self.grid[tuple(position)] == "N":
            reward = -1
        elif self.grid[tuple(position)] == "P":
            reward = 1
        else:
            reward = 0

        if (movable):
            self.current_position = position

        return next_state, reward, is_done, {}

    def reset(self):
        self.current_position = (0, 0)  # Start Position
        return self.current_position

    def render(self):
        for row in range(4):
            for col in range(4):
                if self.current_position == [row, col]:
                    print("X", end=" ")
                else:
                    print(self.grid[row, col], end=" ")
            print("")  # Newline at the end of the row

def compute_value_grid_world(array, state, action, mdp: GridWorldEnv, gamma: float):
    mdp.current_position = state
    step = mdp.step(action)

    next_state = step[0]
    reward = step[1]
    
    
    if (reward == -1 or reward == 1):
        return reward 

    return reward + gamma * array[next_state]

def grid_world_value_iteration(
    env: GridWorldEnv,
    max_iter: int = 1000,
    gamma=1.0,
    theta=1e-5,
) -> np.ndarray:
    """
    Estimation de la fonction de valeur grâce à l'algorithme "value iteration".
    theta est le seuil de convergence (différence maximale entre deux itérations).
    """
    values = np.zeros((4, 4))
    print(f"ENV = {env.observation_space[0].n}")
    

    for i in range(max_iter):
        change = 0
        print(f"===== ITER {i} ======")
        current_values = dp(values)

        for state_i in range(env.observation_space[0].n):
            for state_j in range(env.observation_space[1].n):
                if (env.grid[state_i, state_j] != "F" and env.grid[state_i, state_j] != "S"):
                    continue

                choices = [compute_value_grid_world(values, (state_i, state_j), action, env, gamma) 
                       for action in range(env.action_space.n)]
                
                print(f"==============> for state {state_i}, {state_j}")
                print(f'choices = {choices}\n')
                previous_values = dp(current_values)   
                current_values[state_i, state_j] = max(choices)
                
                # print(previous_values)
                # print(current_values)
                # print (np.abs(current_values[state_i, state_j] - previous_values[state_i, state_j]))
                if (np.abs(current_values[state_i, state_j] - previous_values[state_i, state_j]) > theta):
                    change += 1
        
        values = dp(current_values)
        print(f"{change =}")
        if (change == 0):
            break

    print(f"\n\nfinal values : \n{values}")
    return values
    




def test_grid_world_value_iteration():
    env = GridWorldEnv()

    values = grid_world_value_iteration(env, max_iter=1000, gamma=1.0)
    solution = np.array(
        [
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    assert np.allclose(values, solution)

    values = grid_world_value_iteration(env, max_iter=1000, gamma=0.9)
    solution = np.array(
        [
            [0.81, 0.9, 1.0, 0.0],
            [0.729, 0.0, 0.9, 0.0],
            [0.6561, 0.729, 0.81, 0.729],
            [0.59049, 0.6561, 0.729, 0.6561],
        ]
    )
    assert np.allclose(values, solution)


# Exercice 4: GridWorld avec du bruit
# -----------------------------------
# Ecrire une fonction qui calcule la fonction de valeur pour le GridWorld
# avec du bruit.
# Le bruit est un mouvement aléatoire de l'agent vers sa gauche ou sa droite avec une probabilité de 0.1.


class StochasticGridWorldEnv(GridWorldEnv):
    def _add_noise(self, action: int) -> int:
        prob = random.uniform(0, 1)
        if prob < 0.1:  # 10% chance to go left
            return (action - 1) % 4
        elif prob < 0.1:  # 10% chance to go right
            return (action + 1) % 4
        # 80% chance to go in the intended direction
        return action

    def step(self, action):
        action = self._add_noise(action)
        return super().step(action)

def compute_stochastic(array, state, action, mdp: StochasticGridWorldEnv, gamma: float):
    mdp.current_position = state
    actions = [(action - 1) % 4, (action + 1) % 4, action]
    prob_action = [0.05, 0.05, 0.9]
    
    value = 0
    for action, prob in zip(actions, prob_action):
        step = mdp.step(action)
        next_state = step[0]
        reward = step[1]
        
        
        if (mdp.grid[next_state[0], next_state[1]] == "W"):
            next_state = state

        value += prob * (reward + gamma * array[next_state])


    return value 

def stochastic_grid_world_value_iteration(
    env: StochasticGridWorldEnv,
    max_iter: int = 1000,
    gamma=1.0,
    theta=1e-5,
) -> np.ndarray:
    values = np.zeros((4, 4))
    delta = 0 

    for i in range(max_iter):
        current_values = dp(values)

        for state_i in range(env.observation_space[0].n):
            for state_j in range(env.observation_space[1].n):
                if (env.grid[state_i, state_j] != "F" and env.grid[state_i, state_j] != "S"):
                    continue

                choices = [compute_stochastic(values, (state_i, state_j), action, env, gamma) 
                       for action in range(env.action_space.n)]
                
                previous_values = dp(current_values)   
                current_values[state_i, state_j] = max(choices)
        
                delta = max(delta, (np.abs(current_values[state_i, state_j] - previous_values[state_i, state_j])))
        
        
        
        values = dp(current_values)
        if (delta < theta):
            break


    print(f"\n\nfinal values : \n{values}")
    return values


def test_stochastic_grid_world_value_iteration():
    env = StochasticGridWorldEnv()

    values = stochastic_grid_world_value_iteration(env, max_iter=1000, gamma=1.0)
    solution = np.array(
        [
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    assert np.allclose(values, solution)

    values = stochastic_grid_world_value_iteration(env, max_iter=1000, gamma=0.9)
    solution = np.array(
        [
            [0.73150697, 0.83310665, 0.96151603, 0.0],
            [0.63232473, 0.0, 0.64154523, 0.0],
            [0.54146311, 0.48655038, 0.54726419, 0.47417735],
            [0.47112049, 0.43185906, 0.47417735, 0.41635033],
        ]
    )
    assert np.allclose(values, solution)


# Exercice 3: Evaluation de politique
# -----------------------------------
# Ecrire une fonction qui évalue la politique suivante:
#   - état 0, action 0
#   - état 1, action 0
#   - état 2, action 1


"""
Partie 2 - Programmation dynamique
==================================

Rappel: la programmation dynamique est une technique algorithmique qui
permet de résoudre des problèmes en les décomposant en sous-problèmes
plus petits, et en mémorisant les solutions de ces sous-problèmes pour
éviter de les recalculer plusieurs fois.
"""

# Exercice 1: Fibonacci
# ----------------------
# La suite de Fibonacci est définie par:
#   F(0) = 0
#   F(1) = 1
#   F(n) = F(n-1) + F(n-2) pour n >= 2
#
# Ecrire une fonction qui calcule F(n) pour un n donné.
# Indice: la fonction doit être récursive.


def fibonacci(n: int) -> int:
    """
    Calcule le n-ième terme de la suite de Fibonacci.
    """
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-2) + fibonacci(n-1)


# Tests pour l'exercice 1
@pytest.mark.parametrize(
    "n,expected",
    [
        (0, 0),
        (1, 1),
        (2, 1),
        (3, 2),
        (5, 5),
        (10, 55),
        (20, 6765),
    ],
)
def test_fibonacci(n, expected):
    assert fibonacci(n) == expected


# Exercice 2: Fibonacci avec mémorisation
# ---------------------------------------
# Ecrire une fonction qui calcule F(n) pour un n donné, en mémorisant
# les résultats intermédiaires pour éviter de les recalculer plusieurs
# fois.
# Indice: la fonction doit être récursive.

memo = []
def fibonacci_memo(n: int) -> int:
    """
    Calcule le n-ième terme de la suite de Fibonacci, en mémorisant les
    résultats intermédiaires.
    """
    print(f"==============> n: {n}")
    print(f"==============> memo: {memo}\n")
    if n == 0:
        if len(memo) == 0:
            memo.append(0)
        return 0
    elif n == 1:
        if len(memo) == 1:
            memo.append(1)
        return 1
    else:
        
        if len(memo) == n:
            f1 = memo[n-1]
            f2 = memo[n-2]
            memo.append(f1 + f2)
        else:
            f1 = fibonacci_memo(n-1)
            f2 = fibonacci_memo(n-2)
        return f1 + f2
    


# Tests pour l'exercice 2
@pytest.mark.parametrize(
    "n,expected",
    [
        (0, 0),
        (1, 1),
        (2, 1),
        (3, 2),
        (5, 5),
        (10, 55),
        (20, 6765),
    ],
)
def test_fibonacci_memo(n, expected):
    assert fibonacci_memo(n) == expected


# Exercice 3 : pavage d'un rectangle avec des dominos
# ---------------------------------------------------
# On considère un rectangle de dimensions 3xN, et des dominos de
# dimensions 2x1. On souhaite calculer le nombre de façons de paver le
# rectangle avec des dominos.

# Ecrire une fonction qui calcule le nombre de façons de paver le
# rectangle de dimensions 3xN avec des dominos.
# Indice: trouver une relation de récurrence entre le nombre de façons
# de paver un rectangle de dimensions 3xN et le nombre de façons de
# paver un rectangle de dimensions 3x(N-1), 3x(N-2) et 3x(N-3).


def domino_paving(n: int) -> int:
    """
    Calcule le nombre de façons de paver un rectangle de dimensions 3xN
    avec des dominos.
    """
    a = 0

    if (n % 2 == 1):
        return 0
    elif (n <= 0):
        return 1

    return 4 * domino_paving(n-2) - domino_paving(n-4)


# Tests pour l'exercice 3
@pytest.mark.parametrize(
    "n,expected",
    [
        (1, 0),
        (2, 3),
        (3, 0),
        (4, 11),
        (5, 0),
        (6, 41),
        (7, 0),
        (8, 153),
        (9, 0),
        (10, 571),
    ],
)
def test_domino_paving(n, expected):
    assert domino_paving(n) == expected
