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




def play_and_train(env, agent, iter, trainable=False):
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
        if (trainable):
            loss = agent.train()
            total_loss += 0 if loss is None else loss 

        total_reward += reward
        state = env.reset() if done else next_state
    
    mean_loss = total_loss / iter
    return total_reward, mean_loss




print("========================== DQN ==========================")
agent = DQLearningAgent(
    learning_rate=0.001,
    epsilon_start=1,
    epsilon_end=0.1,
    epsilon_decay_duration=20000,
    gamma=0.99,
    batch_size=16,
    legal_actions=np.arange(env.action_space.n),
    max_memory_size=2000,
    network_update_frequency=1000)

step_per_episode = 1000
final_reward = 0

for i in range(10):
    print(f"Step: {i}")
    if i < 2:
        reward, loss = play_and_train(env, agent, step_per_episode)
    else:
        reward, loss = play_and_train(env, agent, step_per_episode, trainable=True)

    final_reward += reward
    print(f"Loss: {loss} Reward: {reward}, EPS: {agent.epsilon}")

    if (i%20 == 0):
        torch.save(agent.model.state_dict(), "simple_dqn.pt")
        print("Step: {} Reward: {}".format(i, final_reward))
         




    
    