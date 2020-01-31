class LinearValueFunction:
    def __init__(self, step_size):
        self.step_size = step_size

        # Use a tile coding with only a single tiling (i.e. state aggregation):
        # a grid of square tiles
        self.tile_size = 16
        self.w = np.zeros(((BOUNDARY_SOUTH - BOUNDARY_NORTH + self.tile_size) // self.tile_size,
                           (BOUNDARY_EAST - BOUNDARY_WEST + self.tile_size) // self.tile_size,
                           4))

    # Return estimated action value of given state and action
    def value(self, state, action):
        if is_goal_reached(state):
            return 0.0
        return self.w[(state[1] - BOUNDARY_NORTH) // self.tile_size, (state[0] - BOUNDARY_WEST) // self.tile_size, action]

    # Return vector of estimated action values of given state, for each action
    def values(self, state):
        if is_goal_reached(state):
            return np.zeros(4)
        return self.w[(state[1] - BOUNDARY_NORTH) // self.tile_size, (state[0] - BOUNDARY_WEST) // self.tile_size, :]

    # learn with given state, action and target
    def learn(self, state, action, target):
         self.w[(state[1] - BOUNDARY_NORTH) // self.tile_size,
                (state[0] - BOUNDARY_WEST) // self.tile_size,
                action] += (
             self.step_size
             * (target
                - self.w[(state[1] - BOUNDARY_NORTH) // self.tile_size,
                         (state[0] - BOUNDARY_WEST) // self.tile_size,
                         action]))

    # Return estimated state value, based on the estimated action values
    def state_value(self, state):
        return np.max(self.values(state))
