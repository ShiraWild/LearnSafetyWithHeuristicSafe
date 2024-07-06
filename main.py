# imports
import gym
from modified_envs.mountain_car import MountainCarWithCost
from gym import spaces, register
import copy
from ppo import PPO
import numpy as np
import torch
import random
import argparse
import time



# Arguments Declaration

parser = argparse.ArgumentParser()

parser.add_argument("--lr_actor", type=float, default=0.0003,
                    help="learning rate for actor network")
parser.add_argument("--lr_critic", type=float, default=0.001,
                    help="learning rate for critic network")
parser.add_argument("--gamma", type=float, default=0.99,
                    help="gamma for PPO algorithm (adam optimizer)")
# TODO - remove?
parser.add_argument("--K_epochs", type=int, default=80,help = 'K epochs for updating PPO')
parser.add_argument("--eps_clip", type=float, default=0.2,
                    help="clip parameter for PPO")
parser.add_argument("--action_std", type=float, default=0.6,
                    help="action STD for PPO")
parser.add_argument("--max_time_steps", type=int, default=5000,
                    help="max time steps in each run")
parser.add_argument("--max_ep_len", type=int, default=200,
                    help="max episode length.")
parser.add_argument("--update_freq", type=int, default=5,
                    help="update frequency for PPO, after x episodes")
parser.add_argument("--update_stats_saving", type=int, default=2,
                    help="save stats after x episodes")



# TODO - CRITIC params -  change default value here
parser.add_argument("--max_safe_velocity", type=float, default=0.05,
                    help="max safe velocity for cost treshold")
parser.add_argument("--unsafe_tresh_for_masking", type=float, default=0.5,
                    help="max safe treshold for masking")
parser.add_argument("--mc_depth", type=int, default=3,
                    help="determined depth for monte carlo search.")
parser.add_argument("--use_safe_heuristic", type=bool, default=True,
                    help="true - use safe heuristic, false - dont use.")



# aid functions

def register_env(env_id, entry_point):
    register(
        id=env_id,
        entry_point=entry_point,
        kwargs={'max_safe_velocity': args.max_safe_velocity}
    )

def compute_ppo_rewards_and_terminal(env, ppo_selected_action):
    # an aid function, helps to get the 'reward' to be saved in the PPO algorithm
    copied_env = copy.deepcopy(env)
    state, reward, done, info = copied_env.step(ppo_selected_action.item())
    return reward, done

def compute_heuristic_masking(monte_carlo_scores, unsafe_tresh):
    # aid function to compute the masking based on heuristic (MC) scores
    masking_list = [0 if monte_carlo_scores[key] >= unsafe_tresh else 1 for key in monte_carlo_scores]
    # 0 - not safe, 1 - safe
    return masking_list

def monte_carlo_safety_estimate_per_action(env, state, action,  mc_depth):
    # an aid function, returns the safety score for a given state+action based on our heuristic
    total_cost = []
    state, reward, done, info = env.step(action)
    cost = info['cost']
    total_cost.append(cost)
    # Perform actions in the copied environment up to the specified depth, from the given action
    for _ in range(mc_depth):
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        cost = info['cost']
        # 1 - action is safe. 0 - action is not safe
        total_cost.append(cost)
        if done:
            break
    # returns the average cost over all 'depths' per 1 action
    return sum(total_cost)/len(total_cost)

def monte_carlo_safety_estimate(env, state,  mc_depth):
    candidate_actions = {0: None, 1: None, 2: None}
    # work on copied environment and state to avoid changing the original env dynamics
    copied_env = copy.deepcopy(env)
    copied_state = copy.deepcopy(state)
    for action in list(candidate_actions.keys()):
        candidate_actions[action] = monte_carlo_safety_estimate_per_action(copied_env, copied_state, action, mc_depth)
    # return an estimated score 'heuristic' for each score
    # higher = not safe.
    return candidate_actions


# Arguments Parsing
args = parser.parse_args()

# PPO arguments
has_continuous_action_space = False
lr_actor = args.lr_actor
lr_critic = args.lr_critic
gamma = args.gamma
K_epochs = args.K_epochs
eps_clip = args.eps_clip
action_std = args.action_std

# general arguments

max_time_steps = args.max_time_steps
max_ep_len = args.max_ep_len
UPDATE_FREQ = args.update_freq
SAVE_STATS = args.update_stats_saving
env_name = 'MountainCarWithCost-v0'
entry_point = 'modified_envs.mountain_car:MountainCarWithCost'



# critic arguments for experiments

max_safe_velocity = args.max_safe_velocity
unsafe_tresh_for_masking = args.unsafe_tresh_for_masking
mc_depth = args.mc_depth
base_path = "stats/"
use_safe_heuristic = args.use_safe_heuristic



# initializations

# env
register_env(env_name, entry_point)
env = gym.make(env_name)
possible_actions = [0,1,2]


# algorithm
state_dim = np.prod(env.observation_space.shape)
action_dim = env.action_space.n
RL_algorithm = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

# counters or stats lists

amount_of_episodes = 0
time_steps, rewards, costs, episode_lengths = [[] for _ in range(4)]
time_step = 0



while time_step <= max_time_steps:
    # start a new episode with a random action
    random_action = random.choice(possible_actions)
    state, info = env.reset()

    state, reward, done, info = env.step(random_action)
    current_ep_reward = 0
    current_ep_cost = 0
    current_ep_len = 0
    amount_of_episodes += 1
    for t in range(1, max_ep_len + 1):
        time_step += 1
        # get policy probabilities
        ppo_action_probs, ppo_action_log_prob, ppo_selected_action = RL_algorithm.select_action(state)

        # TODO - show Or and Noa - IDK!
        ppo_reward, ppo_terminal = compute_ppo_rewards_and_terminal(env=env, ppo_selected_action=ppo_selected_action)
        RL_algorithm.buffer.rewards.append(torch.tensor(ppo_reward))
        RL_algorithm.buffer.is_terminals.append(torch.tensor(ppo_terminal))


        # use masking for the monte carlo scores. 0 - not safe, 1 - safe
        if use_safe_heuristic:
            # get monte carlo scores - using depth 3
            monte_carlo_scores = monte_carlo_safety_estimate(env, state, mc_depth)
            heuristic_masking = compute_heuristic_masking(monte_carlo_scores,unsafe_tresh_for_masking)
            combined_action_probs = ppo_action_probs * heuristic_masking
            combined_action_probs = list(combined_action_probs)
            selected_action = combined_action_probs.index(max(combined_action_probs))
        else:
            # not using heuristic - the selected action is based on PPO only.
            selected_action = ppo_action_probs.index(max(ppo_action_probs))
        # do the step with the combined select action
        state, reward, done, info = env.step(selected_action)

        cost = info['cost']
        current_ep_len += 1
        current_ep_reward += reward
        current_ep_cost += cost

    time_steps.append(time_step)
    rewards.append(current_ep_reward)
    costs.append(current_ep_cost)
    episode_lengths.append(current_ep_len)

    if amount_of_episodes % UPDATE_FREQ == 0:
        RL_algorithm.update()
    if amount_of_episodes % SAVE_STATS == 0:
        torch.save((time_steps, rewards, costs, episode_lengths), base_path + "stats.log")
