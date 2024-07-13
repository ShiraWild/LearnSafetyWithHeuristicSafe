from gym.envs.classic_control import MountainCarEnv




class MountainCarWithCost(MountainCarEnv):
    def __init__(self, max_safe_velocity):
        super().__init__()
        self.max_safe_velocity = max_safe_velocity

    def get_cost(self, state):
        position, velocity = state
        # TODO - determine max_safe_velocity
        max_safe_velocity = self.max_safe_velocity
        is_safe = velocity <=  max_safe_velocity
        # return 0 if the state is safe, 1 - not safe.
        return 1 - int(is_safe)


    def step(self, action):
        state, reward, done, info = super().step(action)
        # TODO - changed here
        info["cost"] = self.get_cost(state)
        return state, reward, done, info