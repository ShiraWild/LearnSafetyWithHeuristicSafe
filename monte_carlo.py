# monte carlo utilities
import copy



class MonteCarloSearch():
    def __init__(self, mc_depth, unsafe_tresh):
        self.mc_depth = mc_depth
        self.unsafe_tresh = unsafe_tresh
    def compute_heuristic_masking(self, candidate_actions):
        # aid function to compute the masking based on heuristic (MC) scores
        masking_list = [0 if candidate_actions[action_score] >= self.unsafe_tresh else 1 for action_score in candidate_actions]
        # 0 - not safe, 1 - safe
        return masking_list

    def monte_carlo_safety_estimate_per_action(self, env, state, action):
        # an aid function, returns the safety score for a given state+action based on our heuristic
        # another deep copy for each action
        copied_env = copy.deepcopy(env)
        total_cost = []
        state, reward, done, info = copied_env.step(action)
        total_cost.append(info['cost'])
        # Perform actions in the copied environment up to the specified depth, from the given action
        for _ in range(self.mc_depth):
            action = copied_env.action_space.sample()
            state, reward, done, info = copied_env.step(action)
            cost = info['cost']
            total_cost.append(cost)
            if done:
                break
        # returns the average cost over all 'depths' per 1 action
        # bigger cost -> safer
        return sum(total_cost)/len(total_cost)

    def monte_carlo_safety_estimate(self, env, state):
        mc_scores_per_action = {0: None, 1: None}
        # work on copied environment and state to avoid changing the original env dynamics

        for action in list(mc_scores_per_action.keys()):
            mc_scores_per_action[action] = self.monte_carlo_safety_estimate_per_action(env, state, action)
        # return an estimated score 'heuristic' for each score
        # higher ->  safer.
        return mc_scores_per_action

