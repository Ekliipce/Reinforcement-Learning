import gymnasium
import numpy as np
import torch
import matplotlib.pyplot as plt

from preprocess import Preprocess
from deep_q_learning import DQLearningAgent



env = gymnasium.make('ALE/Breakout-v5', render_mode='human')
env = Preprocess(env)

# env.reset()
# final_reward = 0
# for i in range(100):
#     env.render()
#     state, reward, done, _, _ = env.step(env.action_space.sample())
    
#     final_reward += reward
    

#     if (i%100 == 0):
#         print("Step: {} Reward: {}".format(i, final_reward))
    
    
#     if done:
#         env.reset()


# print(state.shape)
# print(state)
# plt.imshow(state, cmap='gray')
# plt.show()

# env.close()




def play_and_train(env, agent, iter):
    state = env.reset()
    total_reward = 0.0
    total_loss = 0.0

    for i in range(iter):
        # 1. Choose an action
        action = agent.get_action(state)

        # 2. Perform the action and observe environment
        next_state, reward, done, _, = env.step(action)
        
        # Update the memory
        agent.update_memory(state, action, reward, next_state, done)

        #3. Train the agent based on the observed environment 
        loss = agent.train()

        total_reward += reward
        total_loss += 0 if loss is None else loss 
        state = env.reset() if done else next_state
    
    mean_loss = total_loss / iter
    return total_reward, mean_loss




print("========================== DQN ==========================")
agent = DQLearningAgent(
    learning_rate=0.001,
    epsilon_start=1,
    epsilon_end=0.1,
    epsilon_decay_duration=1000000,
    gamma=0.99,
    batch_size=32,
    legal_actions=np.arange(env.action_space.n),
    max_memory_size=10000,
    network_update_frequency=10000)

step_per_episode = 500
final_reward = 0

for i in range(100):
    print(f"Step: {i}")
    reward, loss = play_and_train(env, agent, step_per_episode)

    final_reward += reward
    print(f"Loss: {loss} Reward: {reward}")

    if (i%10 == 0):
        torch.save(agent.model.state_dict(), "simple_dqn.pt")
        print("Step: {} Reward: {}".format(i, final_reward))
         




    
    