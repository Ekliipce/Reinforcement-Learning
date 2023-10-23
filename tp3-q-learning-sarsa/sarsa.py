from collections import defaultdict
import random
import typing as t
import numpy as np
import gymnasium as gym


Action = int
State = int
Info = t.TypedDict("Info", {"prob": float, "action_mask": np.ndarray})
QValues = t.DefaultDict[int, t.DefaultDict[Action, float]]


class SarsaAgent:
    def __init__(
        self,
        learning_rate: float,
        epsilon: float,
        gamma: float,
        legal_actions: t.List[Action],
    ):
        """
        SARSA  Agent

        You shoud not use directly self._qvalues, but instead of its getter/setter.
        """
        self.legal_actions = legal_actions
        self._qvalues: QValues = defaultdict(lambda: defaultdict(int))
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma

    def get_qvalue(self, state: State, action: Action) -> float:
        """
        Returns Q(state,action)
        """
        return self._qvalues[state][action]

    def set_qvalue(self, state: State, action: Action, value: float):
        """
        Sets the Qvalue for [state,action] to the given value
        """
        self._qvalues[state][action] = value

    def get_value(self, state: State) -> float:
        """
        Compute your agent's estimate of V(s) using current q-values
        V(s) = max_a Q(s, a) over possible actions.
        """
        qvalues = [self.get_qvalue(state, action) for action in self.legal_actions]
        value = np.max(qvalues)
        return value

    def update(
        self, state: State, action: Action, reward: t.SupportsFloat, next_state: State
    ):
        """
        You should do your Q-Value update here (s'=next_state):
           TD_target(s') = R(s, a) + gamma * Q(s', a')
           TD_error(s', a) = TD_target(s') - Q(s, a)
           Q_new(s, a) := Q(s, a) + alpha * TD_error(s', a)
        """
        
        old_q = self.get_qvalue(state, action)
        next_action = self.get_action(state)
        future_q = self.get_qvalue(next_state, next_action)
       

        temporal_difference_target = reward + self.gamma * future_q
        temporal_difference_error = temporal_difference_target - old_q
        q_value = old_q + self.learning_rate * temporal_difference_error

        self.set_qvalue(state, action, q_value)

    def get_best_action(self, state: State) -> Action:
        """
        Compute the best action to take in a state (using current q-values).
        """
        possible_q_values = [
            self.get_qvalue(state, action) for action in self.legal_actions
        ]
        index = np.argmax(possible_q_values)
        best_action = self.legal_actions[index]
        return best_action

    def get_action(self, state: State) -> Action:
        """
        Compute the action to take in the current state, including exploration.
        """
       
        p = np.random.uniform(0, 1)

        if (p <= self.epsilon): #Exploration
            action = random.choice(self.legal_actions)
        else : #Exploitation
            action = self.get_best_action(state)

        return action