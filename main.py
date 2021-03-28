import gym
import numpy as np
import random as rnd

env = gym.make("Taxi-v3").env

action_space_size = env.action_space.n
obs_space_size = env.observation_space.n
q_table = np.zeros((obs_space_size, action_space_size))


alpha = 0.1
gamma = 0.6
epsilon = 0.1

all_epochs = []
all_penalties = []

for i in range(1, 100001):
    state = env.reset()
    epochs, penalties, rewards = 0, 0, 0
    done = False

    while not done:
        if epsilon > rnd.uniform(0, 1):
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        

        next_state, reward, done, info = env.step(action)

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        new_values = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_values

        if reward == -10:
            penalties += 1
        
        state = next_state

        epochs += 1

    if i % 100 == 0:
        print(f"Episode: {i}")

        
        






