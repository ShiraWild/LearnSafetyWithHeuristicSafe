import matplotlib.pyplot as plt
import torch


time_steps, rewards, costs, episode_lengths = torch.load("stats/stats.log")


# Calculate rewards per episode_length

def plot_rewards(time_steps, rewards, episode_lengths):
    plt.figure(figsize=(10, 6))
    rewards_per_episode_length = [reward / length for reward, length in zip(rewards, episode_lengths)]
    plt.plot(time_steps, rewards_per_episode_length, marker='o', linestyle='-', color='b',
             label='Rewards / Episode Length')
    plt.xlabel('Time Steps')
    plt.ylabel('Rewards / Episode Length')
    plt.title('Time Steps vs. Rewards / Episode Length')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Plotting

def plot_costs(time_steps, costs, episode_lengths):
    plt.figure(figsize=(10, 6))
    costs_per_episode_length = [cost / length for cost, length in zip(costs, episode_lengths)]
    plt.plot(time_steps, costs_per_episode_length, marker='o', linestyle='-', color='b',
             label='Costs / Episode Length')
    plt.xlabel('Time Steps')
    plt.ylabel('Costs / Episode Length')
    plt.title('Time Steps vs. Costs / Episode Length')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


#plot_rewards(time_steps,rewards,episode_lengths)
plot_costs(time_steps, costs, episode_lengths)