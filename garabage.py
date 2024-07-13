# main.py

import gym
from modified_envs.cart_pole import CartPoleWithCost

# Define a unique ID for your environment
env_id = 'CartPoleWithCost-v1'

# Register your custom environment with Gym
gym.envs.register(
    id=env_id,
    entry_point='modified_envs.cart_pole:CartPoleWithCost',
    max_episode_steps=200,  # Adjust as per your needs
    reward_threshold=195.0,  # Adjust as per your needs
)

# Create an instance of your custom environment
env = gym.make(env_id)

# Now you can use 'env' like any other Gym environment
observation = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    if done:
        observation = env.reset()

env.close()  # Remember to close the environment when done
