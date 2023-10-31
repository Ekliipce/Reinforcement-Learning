import gymnasium as gym
import numpy as np
import torch
import torchvision.transforms as T

from PIL import Image
from gymnasium import ObservationWrapper


class Preprocess(ObservationWrapper):
    # def __init__(self, env, shape=(84, 84), history_length=4):
    #     super().__init__(env)
    #     self.shape = shape
        
    #     self.transform = T.Compose([
    #         T.Grayscale(num_output_channels=1),
    #         T.Resize(self.shape),
    #         T.ToTensor()
    #     ])

    # def observation(self, observation):
    #     observation = Image.fromarray(observation)
    #     observation = self.transform(observation).squeeze(0)
    #     return observation
    

    def __init__(self, env, shape=(84, 84), history_length=4):
        super().__init__(env)
        self.shape = shape
        self.history = np.zeros((history_length, shape[0], shape[1]), dtype=np.float32)
        
        self.transform = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.Resize(self.shape),
            T.ToTensor()
        ])

    def reset(self, **kwargs):
        observation = super().reset(**kwargs)
        for _ in range(self.history.shape[0]):
            self.push_to_history(observation[0])
        return self.get_stacked_observations()

    def push_to_history(self, observation):
        observation = self.transform(Image.fromarray(observation)).squeeze(0)
        self.history[:-1] = self.history[1:]
        self.history[-1] = observation.numpy()

    def get_stacked_observations(self):
        return np.moveaxis(self.history, 0, -1)  # move the channel axis to the end

    def step(self, action):
        observation, reward, done, info, _ = super().step(action)
        self.push_to_history(observation)
        return self.get_stacked_observations(), reward, done, info

    def observation(self, observation):
        return observation