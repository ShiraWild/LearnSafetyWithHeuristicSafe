from modified_envs.cart_pole import *

class CartPoleWithCost(CartPoleEnv):
    def __init__(self, param_ranges):
        super().__init__()
        default_parameter_values = {}
        for param, param_range in param_ranges.items():
            default_parameter_values[param] = self.np_random.uniform(low=param_range['range'][0], high=param_range['range'][1])
        self.update_params(default_parameter_values)

    def get_cost(self, state):
        pole_angle = state[2]
        cart_position = state[0]
        if abs(pole_angle) >= self.theta_threshold_radians or abs(cart_position) >= self.x_threshold:
            return 1
        else:
            return 0

    # override step function to add cost to info

    def step(self, action):
        state, reward, done, _, info = super().step(action)
        # TODO - changed here
        info["cost"] = self.get_cost(state)
        return state, reward, done, info