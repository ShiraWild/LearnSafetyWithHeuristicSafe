# imports
import copy
import gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from IPython import display
import argparse
from gym import spaces, register
from modified_envs.cart_pole import CartPoleWithCost
from monte_carlo import MonteCarloSearch

# arguments


parser = argparse.ArgumentParser()

# TODO - CRITIC params -  change default value here
parser.add_argument("--cart_velocity_tresh", type=float, default=2.0,
                    help="max safe cart velocity for cost treshold")

parser.add_argument("--unsafe_tresh", type=float, default=0.5,
                    help="safety_treshold for masking")
parser.add_argument("--mc_depth", type=int, default=5,
                    help="determined depth for monte carlo search.")
parser.add_argument("--use_safe_heuristic", type=bool, default=True,
                    help="True - use safe heuristic, False - don't use.")


args = parser.parse_args()

#cart_velocity_tresh = args.cart_velocity_tresh
unsafe_tresh = args.unsafe_tresh
mc_depth = args.mc_depth
base_path = "stats/"
use_safe_heuristic = args.use_safe_heuristic


# set up the environment

env_id = 'CartPoleWithCost-v0'

# Register custom env with Gym
gym.envs.register(
    id=env_id,
    entry_point='modified_envs.cart_pole:CartPoleWithCost')

env = gym.make(env_id)


# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

# set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device is set to: {device}")


# DQN Pytorch Algorithm - https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))




cart_velocity_tresh = 2.0



class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4


# set up env

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state = env.reset()
n_observations = len(state)

# define policy network and target network (DDQN)
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0


def epsilon_greedy_select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    exploitation_based = False
    if sample > eps_threshold:
        # exploitation
        with torch.no_grad():
            # compute the Q-Values for all actions given current state
            # take .max(1) - the maximum Q-values across all actions

            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            exploitation_based = True
            return policy_net(state).max(1).indices.view(1, 1), exploitation_based, policy_net(state)
    else:
        # exploration
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long), exploitation_based, None


episode_rewards = []

episode_costs = []

def plot_stats(show_result=False):
    plt.figure(1)
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    costs_t = torch.tensor(episode_costs, dtype=torch.float)

    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.subplot(2,1,1)
    plt.xlabel('Episode')
    plt.ylabel('Rewards (Episode Duration)')
    plt.plot(rewards_t.numpy(), label = "Rewards")
    # Take 100 episode averages and plot them too
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy(), label = "100-episode Moving Avg")
    plt.legend()

    # Plot costs
    plt.subplot(2, 1, 2)
    plt.xlabel('Episode')
    plt.ylabel('Costs')
    plt.plot(costs_t.numpy(), label='Costs')
    if len(costs_t) >= 100:
        cost_means = costs_t.unfold(0, 100, 1).mean(1).view(-1)
        cost_means = torch.cat((torch.zeros(99), cost_means))
        plt.plot(cost_means.numpy(), label='100-episode Moving Avg')
    plt.legend()

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state).to(device)
    action_batch = torch.cat(batch.action).to(device)
    reward_batch = torch.cat(batch.reward).to(device)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()




num_episodes = 1000

TRUNCATION_TRESH = 200

mc_search = MonteCarloSearch(mc_depth = mc_depth, unsafe_tresh = unsafe_tresh)

for i_episode in range(num_episodes):
    truncated = False
    episode_cost = 0
    # plots every 100 episodes
    if i_episode % 100 ==0:
        if i_episode != 0:
            print(f"Finished {i_episode} episodes. Continue training.. ")
            plot_stats()
    # Initialize the environment and get its state
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        if (t+1) == TRUNCATION_TRESH:
            truncated = True
        # epsilon greedy
        selected_action, exploitation_based, dqn_scores = epsilon_greedy_select_action(state)
        if exploitation_based:
            # can use monte carlo
            copied_env = copy.deepcopy(env)
            copied_state = copy.deepcopy(state)
            mc_scores_per_action = mc_search.monte_carlo_safety_estimate(copied_env, copied_state)
            masked_actions = mc_search.compute_heuristic_masking(mc_scores_per_action)
            unsafe_actions = masked_actions.count(0)
            # at least one action is not safe
            if unsafe_actions == 1:
                unsafe_action = [idx for idx, mask in enumerate(masked_actions) if mask == 0][0]
                # take the left action.
                selected_action = torch.tensor([[1 - unsafe_action]], device=device)
        action = selected_action
        observation, reward, terminated, info = env.step(action.item())
        cost = info['cost']
        episode_cost += cost
        reward = torch.tensor([reward], device=device)
        done = terminated

        if terminated or truncated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done or truncated:
            episode_len = t+1
            episode_rewards.append(episode_len)
            episode_costs.append(episode_cost/episode_len)
            break

print('Complete')
plot_stats(show_result=True)
plt.ioff()
plt.show()
