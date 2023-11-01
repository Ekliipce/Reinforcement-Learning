import random
import numpy as np
import gymnasium as gym
import typing as t
import torch

from collections import deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DQNModel(torch.nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQNModel, self).__init__()
        layer_shape = [4, 32, 64]
        stride_shape = [4, 2, 1]

        self.input_shape = input_shape
        self.num_actions = num_actions
        self.conv1 = torch.nn.Conv2d(layer_shape[0], 32, kernel_size=(3, 3), stride=stride_shape[0])
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=(3, 3), stride=stride_shape[1])
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=(3, 3), stride=stride_shape[2])

        def conv_size(size, kernel, stride):
            return (size - kernel) // stride + 1
        
        width = 84
        width = conv_size(conv_size(conv_size(width, 3, stride_shape[0]),
                         3, stride_shape[1]),
                           3, stride_shape[2])
        height = width
        
        linear_input_size = width * height * 64

        self.fc1 = torch.nn.Linear(linear_input_size, 512) 
        self.fc2 = torch.nn.Linear(512, num_actions) 

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.nn.functional.relu(self.fc1(x))
        return self.fc2(x)

class DQLearningAgent:
    def __init__(
        self,
        learning_rate : float,
        epsilon_start : float,
        epsilon_end : float,
        epsilon_decay_duration : int,
        gamma : float,
        batch_size : int,
        legal_actions : t.List[int],
        max_memory_size : int,
        network_update_frequency,
    ):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.legal_actions = legal_actions

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_duration = epsilon_decay_duration
        self.epsilon = epsilon_start
        self.steps_done = 0

        self.memory_play = deque(maxlen=max_memory_size)


        self.model = DQNModel((84, 84), len(legal_actions)).to(device)
        self.target_model = DQNModel((84, 84), len(legal_actions)).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.network_update_frequency = network_update_frequency

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = torch.nn.MSELoss()



    def train(self):
        self.model.train()
        if len(self.memory_play) <= self.batch_size:
            return None

        # Update the target network each C steps
        if self.steps_done % self.network_update_frequency == 0:
            self.update_target_network()
        
        states, actions, rewards, next_states, dones = self.sample_experiences(self.batch_size)
        
        # Compute Q(s_t, a) using the target network
        with torch.no_grad():
            q_values_next = self.target_model(next_states)
        max_q_values_next, _ = q_values_next.max(dim=1)
        target_q_values = rewards + (1 - dones) * self.gamma * max_q_values_next

        # Forwards pass
        predicted_q_values = self.model(states)
        q_values = torch.gather(predicted_q_values, dim=1, index=actions.unsqueeze(-1)).squeeze(-1)
        
        # Clamp loss between -1 and 1 for more stability 
        td_errors = target_q_values - q_values
        clipped_td_errors = torch.clamp(td_errors, -1, 1)
        
        loss = self.loss_fn(q_values, q_values + clipped_td_errors)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()



    def sample_experiences(self, batch_size):
        index = np.random.choice(len(self.memory_play), batch_size, replace=False)
        batch = [self.memory_play[i] for i in index]
        states, actions, rewards, next_states, dones = zip(*batch)

        #Create Batch (32, 4, 84, 84)
        states = torch.cat(states, dim=0)
        next_states = torch.cat(next_states, dim=0)

        #Tuple to Tensor
        actions = torch.tensor(actions, dtype=torch.int64).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)

        return states, actions, rewards,next_states, dones



    def convert_tottensor(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).to(device) 
        state_tensor = state_tensor.permute(2, 0, 1).unsqueeze(0) 
        return state_tensor
    


    def update_memory(self, state, action, reward, next_state, done):
        state = self.convert_tottensor(state)
        next_state = self.convert_tottensor(next_state)
        self.memory_play.append((state, action, reward, next_state, done))



    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())



    def update_epsilon(self):
        diff = self.epsilon_start - self.epsilon_end  
        decay =  np.exp(-1. * self.steps_done / self.epsilon_decay_duration)
        
        self.epsilon = self.epsilon_end + diff * decay


  
    def get_action(self, state : np.ndarray) -> int:
        self.steps_done += 1
        self.update_epsilon()

        if np.random.rand() <= self.epsilon:
            return random.choice(self.legal_actions)
        else:
            q_values = self.model(self.convert_tottensor(state))
            return torch.argmax(q_values[0])
        
        