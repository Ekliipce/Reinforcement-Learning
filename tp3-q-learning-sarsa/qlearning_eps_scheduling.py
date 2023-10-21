import random
import numpy as np
from qlearning import QLearningAgent, State, Action


class QLearningAgentEpsScheduling(QLearningAgent):
    def __init__(
        self,
        *args,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 10000,
        **kwargs,
    ):
        """
        Q-Learning Agent with epsilon scheduling

        You shoud not use directly self._qvalues, but instead of its getter/setter.
        """
        super().__init__(*args, **kwargs)
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.timestep = 0

    def reset(self):
        """
        Reset epsilon to the start value.
        """
        self.epsilon = self.epsilon_start
        self.timestep = 0

    def get_action(self, state: State, decay= 0.005) -> Action:
        """
        Compute the action to take in the current state, including exploration.

        Exploration is done with epsilon-greey. Namely, with probability self.epsilon,
          we should take a random action, and otherwise the best policy action (self.get_best_action).

        Note: To pick randomly from a list, use random.choice(list).
              To pick True or False with a given probablity, generate uniform number in [0, 1]
              and compare it with your probability
        """
        p = np.random.uniform(0, 1)

        if (p <= self.epsilon): #Exploration
            action = random.choice(self.legal_actions)
        else : #Exploitation
            action = self.get_best_action(state)
        
        if (self.epsilon > self.epsilon_end):
            self.epsilon = self.epsilon_start / (1 + (10000 - self.epsilon_decay_steps) * decay)
            self.epsilon_decay_steps -= 1
            
        return action
