"""
Dans ce TP, nous allons implémenter un agent qui apprend à jouer au jeu Taxi-v3
de OpenAI Gym. Le but du jeu est de déposer un passager à une destination
spécifique en un minimum de temps. Le jeu est composé d'une grille de 5x5 cases
et le taxi peut se déplacer dans les 4 directions (haut, bas, gauche, droite).
Le taxi peut prendre un passager sur une case spécifique et le déposer à une
destination spécifique. Le jeu est terminé lorsque le passager est déposé à la
destination. Le jeu est aussi terminé si le taxi prend plus de 200 actions.

Vous devez implémenter un agent qui apprend à jouer à ce jeu en utilisant
les algorithmes Q-Learning et SARSA.

Pour chaque algorithme, vous devez réaliser une vidéo pour montrer que votre modèle fonctionne.
Vous devez aussi comparer l'efficacité des deux algorithmes en termes de temps
d'apprentissage et de performance.

A la fin, vous devez rendre un rapport qui explique vos choix d'implémentation
et vos résultats (max 1 page).
"""

import typing as t
import gymnasium as gym
import numpy as np
from qlearning import QLearningAgent
from qlearning_eps_scheduling import QLearningAgentEpsScheduling
from gym.wrappers.monitoring import video_recorder
from sarsa import SarsaAgent


env = gym.make("Taxi-v3", render_mode="rgb_array")
n_actions = env.action_space.n  # type: ignore


#################################################
# 1. Play with QLearningAgent
#################################################

agent = QLearningAgent(
    learning_rate=0.2, epsilon=0.1, gamma=0.99, legal_actions=list(range(n_actions))
)


def play_and_train(env: gym.Env,
                   agent: QLearningAgent,
                   t_max=int(1e4),
                   create_movie=False,
                   filename=None,
                   long_train=False) -> float:
    """
    This function should
    - run a full game, actions given by agent.getAction(s)
    - train agent using agent.update(...) whenever possible
    - return total rewardb
    """
    total_reward: t.SupportsFloat = 0.0
    s, _ = env.reset()
    if (create_movie):
        vid = video_recorder.VideoRecorder(env, path=f"./movie/{filename}")

    for _ in range(t_max):
        if (create_movie):
            env.render()
            vid.capture_frame()

        # Get agent to pick action given state s
        a = agent.get_action(s)

        next_s, r, done, _, _ = env.step(a)
        total_reward += r 

        # Train agent for state s
        agent.update(s, a, r, next_s)
        s = next_s

        if (done):
            s, _ = env.reset()
            if (not long_train): break

    if (create_movie):
        vid.close()

    return total_reward

print ("================ QLearningAgent ======================")

play_and_train(env, agent, create_movie=True, filename="taxi_notrain.mp4")

rewards = []
for i in range(1000):
    rewards.append(play_and_train(env, agent))

    ## if you want to train more, test long_train parameter
    ##rewards.append(play_and_train(env, agent, 700, long_train=True))
    if i % 100 == 0:
        print("mean reward", np.mean(rewards[-100:]))

env.render()

assert np.mean(rewards[-100:]) > 0.0

# TODO: créer des vidéos de l'agent en action
play_and_train(env, agent, create_movie=True, filename="taxi_qlearning.mp4")


#################################################
# 2. Play with QLearningAgentEpsScheduling
#################################################


print ("================ QLearningAgentEpsScheduling ======================")


agent = QLearningAgentEpsScheduling(
    learning_rate=0.2, epsilon=0.15, gamma=0.99, legal_actions=list(range(n_actions))
)
rewards = []
for i in range(1000):
    rewards.append(play_and_train(env, agent))

    ## if you want to train more, test long_train parameter
    ##rewards.append(play_and_train(env, agent, 700, long_train=True))
    if i % 100 == 0:
        print("mean reward", np.mean(rewards[-100:]))

assert np.mean(rewards[-100:]) > 0.0

# TODO: créer des vidéos de l'agent en action
play_and_train(env, agent, create_movie=True, filename="taxi_qlearning_epsscheduling.mp4")



####################
# 3. Play with SARSA
####################


agent = SarsaAgent(learning_rate=0.2, epsilon=0.05, gamma=0.99, legal_actions=list(range(n_actions)))

print ("================ SARSA ======================")


rewards = []
for i in range(1000):
    rewards.append(play_and_train(env, agent, 700, long_train=True))
    if i % 100 == 0:
        print("mean reward", np.mean(rewards[-100:]))

        
play_and_train(env, agent, create_movie=True, filename="taxi_sarsa.mp4")



"""
        Conclusion :
            If we train with option long_train (reset when the game is finished)
            and 700 steps * 1000 iters we obtain : 
            
            Q-Learning (lr=0.2, eps=0.1, gamma=0.99)
                near 115 of rewards

            Q-learning (lr=0.2, eps=0.15, gamma=0.99)
                near 250 of rewards

            SARSA (lr=0.2, eps=0.05, gamma=0.99)
                near 180 of rewards

"""