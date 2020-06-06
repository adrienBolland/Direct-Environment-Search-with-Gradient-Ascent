class Environment:
    def __init__(self, system_nn):
        self.system_nn = system_nn

        self._state = None

    def reset(self, parallel_trajectories):
        self._state = self.system_nn.initial_state(parallel_trajectories).clone()

        return self

    @property
    def state(self):
        return self._state.clone()

    def step(self, one_hot_actions):
        next_states, disturbances, reward, actions = self.system_nn(self._state, one_hot_actions)
        self._state = next_states
        return next_states.clone(), disturbances, reward, actions

