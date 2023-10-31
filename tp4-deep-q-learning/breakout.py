import gymnasium
import numpy as np
# Visualiser l'état initial
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

    for i in range(iter):
        print("step: {}".format(i))
        print(state.shape)
        # 1. Choose an action
        action = agent.get_action(state)

        print("action chosed")

        # 2. Perform the action and observe environment
        next_state, reward, done, _, = env.step(action)
        print("action done")
        # Update the memory
        agent.update_memory(state, action, reward, next_state, done)

        print("memory updated")
        #3. Train the agent based on the observed environment 
        agent.train()
        print("agent trained")  

        total_reward += reward
        state = env.reset() if done else next_state
    
    return total_reward




print("========================== DQN ==========================")
agent = DQLearningAgent(0.01, 0.1, 0.99, 32, np.arange(env.action_space.n))

final_reward = 0
for i in range(100):
    play_and_train(env, agent, 200)
    

    # if (i%100 == 0):
    #     print("Step: {} Reward: {}".format(i, final_reward))
    