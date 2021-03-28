import gym
import numpy as np
import random as rnd

env = gym.make("Taxi-v3").env
env.reset()
env.render()

print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))

action_space_size = env.action_space
obs_space_size = env.observation_space

q_table = np.zeros((obs_space_size, action_space_size))





