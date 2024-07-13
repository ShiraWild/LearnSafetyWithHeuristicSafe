# imports
import gym
from stable_baselines3 import PPO
import numpy as np
import imageio

# set-up the training environment
env = gym.make('MountainCar-v0')


# Implement actor critic, using a multi-layer perceptron (2 layers of 64) in the pre-specified environment
model = PPO("MlpPolicy", env=env, n_steps=16, gae_lambda=0.98, gamma=0.99, n_epochs=4, ent_coef=0.0, verbose=1)

# return a trained model that is trained over 10,000 timesteps
model.learn(total_timesteps=float(1e6))
model.save("ppo_mountaincar")
