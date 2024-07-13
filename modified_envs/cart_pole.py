from gym.envs.classic_control import CartPoleEnv
import numpy as np

class CartPoleWithCost(CartPoleEnv):
    def __init__(self):
        super().__init__()

    """
    # distances cost
    def get_cost(self, state):
        cart_position, cart_velocity, pole_angle, pole_velocity = state

        # Calculate distance from termination conditions
        distance_from_x_term = abs(abs(cart_position) - abs(self.x_threshold))
        distance_from_theta_term = abs(abs(pole_angle) - abs(self.theta_threshold_radians))
        # the higher the total distances -> the safer the state is (distance from 'unsafe behavior')
        total_distances = distance_from_x_term + distance_from_theta_term
        # TODO - Think here together with Noa and Or if we want to take the -1 * total_distances (opposite loss)
        total_cost =  total_distances
        # the higher cost -> more safe (more distance).
        return total_cost
    # override step function to add cost to info
    """

    def get_cost(self, state):
        cart_position, cart_velocity, pole_angle, pole_velocity = state
        safety_x_treshold = self.x_threshold - 1
        safety_theta_threshold_radians = self.theta_threshold_radians - 0.1
        cost =  cart_position < -safety_x_treshold \
                or cart_position > safety_x_treshold\
                or pole_angle < -safety_theta_threshold_radians \
                or pole_angle > safety_theta_threshold_radians
        # cost 1 = not safe.
        return cost
    # override step function to add cost to info

    def step(self, action):
        state, reward, done,  info = super().step(action)
        # TODO - changed here
        info["cost"] = self.get_cost(state)
        return state, reward, done, info