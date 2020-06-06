import torch

from systems.System import System


class SystemWrapper(System):
    def __init__(self, sys):
        super(SystemWrapper, self).__init__()
        self.sys = sys
        self.observation_space_size = self.sys.observation_space_size
        self.action_space_size = self.sys.action_space_size
        self.env_min = self.sys.env_min
        self.env_max = self.sys.env_max
        self.act_min = self.sys.act_min
        self.act_max = self.sys.act_max
        self.device = self.sys.device

    @classmethod
    def class_name(cls):
        return cls.__name__

    def reward(self, states, actions, disturbances):
        """ reward function rho(s_t, a_t, xi_t) -> r_t"""
        return self.sys.reward(states, actions, disturbances)

    def dynamics(self, states, actions, disturbances):
        """ dynamics f(s_t, a_t, xi_t) -> s_t+1 """
        return self.sys.dynamics(states, actions, disturbances)

    def disturbance(self, states, actions):
        """ disturbance distribution P_xi(.|s_t, a_t)
         returns a torch.distribution object """
        return self.sys.disturbance(states, actions)

    def render(self, states, actions, dist, rewards, num_trj):
        return self.sys.render(states, actions, dist, rewards, num_trj)

    def initial_state(self, number_trajectories):
        """ samples 'number_trajectories' initial states from P_0
         returns a tensor of shape ('number_trajectories', |S|) """
        init_state = self.sys.initial_state(number_trajectories)
        self.env_min = self.sys.env_min
        self.env_max = self.sys.env_max
        self.act_min = self.sys.act_min
        self.act_max = self.sys.act_max
        return init_state

    def forward(self, state, action):
        disturbances = self.disturbance(state, action).sample()

        reward = self.reward(state, action, disturbances)

        next_states = self.dynamics(state, action, disturbances)
        return next_states, disturbances, reward, action

    def project_parameters(self):
        """ performs parameters projection """
        return self.sys.project_parameters()

    def reset_parameters(self):
        """ reset the parameters (require gradient tensors) """
        return self.sys.reset_parameters()

    def parameters_dict(self):
        """ returns a dictionary mapping the parameters' name to their values """
        return self.sys.parameters_dict()

    def stabilization_estimate(self, states_batch):
        """ From a batch of states series of shape (:, T, |S|) assess the performance of the controller """
        return self.sys.stabilization_estimate(states_batch)

    def initialize(self, **kwargs):
        """ instantiate an object from a dictionary """
        return self.sys.initialize(**kwargs)

    def sample_action(self, state):
        return self.sys.sample_action(state)

    @property
    def unwrapped(self):
        return self.sys.unwrapped


class MinMaxScaleStates(SystemWrapper):
    def __init__(self, sys, min_value=-1, max_value=1):
        super(MinMaxScaleStates, self).__init__(sys)
        self.min_value = min_value
        self.max_value = max_value
        self.env_min = torch.full_like(self.sys.env_min, min_value, device=self.device)
        self.env_max = torch.full_like(self.sys.env_max, max_value, device=self.device)

    def down_scale_state(self, state_scaled):
        X_std = (state_scaled - self.env_min) / (self.env_max - self.env_min)
        state_unsc = X_std * (self.sys.env_max - self.sys.env_min) + self.sys.env_min
        return state_unsc

    def up_scale_state(self, state_unscaled):
        X_std = (state_unscaled - self.sys.env_min) / (self.sys.env_max - self.sys.env_min)
        state_scaled = X_std * (self.env_max - self.env_min) + self.env_min
        return state_scaled

    def reward(self, states, actions, disturbances):
        """ reward function rho(s_t, a_t, xi_t) -> r_t"""
        states = self.down_scale_state(states)
        return self.sys.reward(states, actions, disturbances)

    def dynamics(self, states, actions, disturbances):
        """ dynamics f(s_t, a_t, xi_t) -> s_t+1 """
        states = self.down_scale_state(states)
        return self.up_scale_state(self.sys.dynamics(states, actions, disturbances))

    def disturbance(self, states, actions):
        """ disturbance distribution P_xi(.|s_t, a_t)
         returns a torch.distribution object """
        states = self.down_scale_state(states)
        return self.sys.disturbance(states, actions)

    def initial_state(self, number_trajectories):
        """ samples 'number_trajectories' initial states from P_0
         returns a tensor of shape ('number_trajectories', |S|) """
        init_state = self.up_scale_state(self.sys.initial_state(number_trajectories))
        self.env_min = torch.full_like(self.sys.env_min, self.min_value, device=self.device)
        self.env_max = torch.full_like(self.sys.env_max, self.max_value, device=self.device)
        return init_state


class MinMaxScaleActions(SystemWrapper):
    def __init__(self, sys, min_value=-1, max_value=1):
        super(MinMaxScaleActions, self).__init__(sys)
        self.min_value = min_value
        self.max_value = max_value
        # self.act_min = torch.tensor((), device=self.device).new_full((self.sys.action_space_size,), self.min_value)
        # self.act_max = torch.tensor((), device=self.device).new_full((self.sys.action_space_size,), self.max_value)
        self.act_min = torch.tensor((min_value), device=self.device)
        self.act_max = torch.tensor((max_value), device=self.device)

    def down_scale_action(self, action_scaled):
        X_std = (action_scaled - self.act_min) / (self.act_max - self.act_min)
        action_unsc = X_std * (self.sys.act_max - self.sys.act_min) + self.sys.act_min
        return action_unsc

    def sample_action(self, state):
        action = torch.empty((state.shape[0], self.act_min.shape[0]), device=self.device).uniform_(self.min_value,
                                                                                                   self.max_value)
        return action

    def reward(self, states, actions, disturbances):
        actions = self.down_scale_action(actions)
        return self.sys.reward(states, actions, disturbances)

    def dynamics(self, states, actions, disturbances):
        actions = self.down_scale_action(actions)
        return self.sys.dynamics(states, actions, disturbances)

    def disturbance(self, states, actions):
        actions = self.down_scale_action(actions)
        return self.sys.disturbance(states, actions)

    def initial_state(self, number_trajectories):
        """ samples 'number_trajectories' initial states from P_0
         returns a tensor of shape ('number_trajectories', |S|) """
        init_state = self.sys.initial_state(number_trajectories)
        self.env_min = self.sys.env_min
        self.env_max = self.sys.env_max
        self.act_min = torch.tensor((self.min_value), device=self.device)
        self.act_max = torch.tensor((self.max_value), device=self.device)
        return init_state


class RewardScaling(SystemWrapper):
    def __init__(self, sys, min_value=0, max_value=1, min_rew=-5000, max_rew=0.):
        super(RewardScaling, self).__init__(sys)
        self.min_value = min_value
        self.max_value = max_value
        self.min_rew = min_rew
        self.max_rew = max_rew

    def reward(self, states, actions, disturbances):
        rew = self.sys.reward(states, actions, disturbances)
        return self.up_scale_reward(rew)

    def up_scale_reward(self, rew_unscaled):
        X_std = (rew_unscaled - self.min_rew) / (self.max_rew - self.min_rew)
        rew_scaled = X_std * (self.max_value - self.min_value) + self.min_value
        return rew_scaled

    def down_scale_reward(self, rew_scaled):
        X_std = (rew_scaled - self.min_value) / (self.max_value - self.min_value)
        rew_unscaled = X_std * (self.max_rew - self.min_rew) + self.min_rew
        return rew_unscaled


class RewardExpScaling(SystemWrapper):
    def __init__(self, sys):
        super(RewardExpScaling, self).__init__(sys)

    def reward(self, states, actions, disturbances):
        rew = self.sys.reward(states, actions, disturbances)
        return self.up_scale_reward(rew)

    def up_scale_reward(self, rew_unscaled, scale_constant=10000.):
        return torch.exp(rew_unscaled / scale_constant)