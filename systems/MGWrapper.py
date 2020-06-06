import torch

from systems.SystemWrapper import SystemWrapper


class ScaleBatteryActions(SystemWrapper):
    def __init__(self, sys, min_value=-1, max_value=1):
        super(ScaleBatteryActions, self).__init__(sys)
        self.min_value = min_value
        self.max_value = max_value
        self.act_min = torch.tensor((), device=self.device).new_full((self.sys.action_space_size,), self.min_value)
        self.act_max = torch.tensor((), device=self.device).new_full((self.sys.action_space_size,), self.max_value)

    def scale_action(self, action_scaled, state):
        soc, h, pv, dem, lim_ch, lim_dis = state.split(1, dim=1)
        p_b, p_g = action_scaled.split(1, dim=1)
        p_dis = torch.clamp(
            (2 / (self.max_value - self.min_value) * (p_b - (self.max_value + self.min_value) / 2)) * lim_dis, min=0)
        p_ch = torch.clamp(
            (2 / (self.max_value - self.min_value) * (p_b - (self.max_value + self.min_value) / 2)) * lim_ch, max=0)
        p_bat = torch.add(p_dis, p_ch)
        p_gen = (1 / (self.max_value - self.min_value) * (p_g - self.min_value)) * self.unwrapped.gen_size
        return torch.cat((p_bat, p_gen), dim=1)

    def sample_action(self, state):
        action = torch.empty((state.shape[0], self.act_min.shape[0]), device=self.device).uniform_(self.min_value,
                                                                                                   self.max_value)
        return action

    def reward(self, states, actions, disturbances):
        actions = self.scale_action(actions, states)
        return self.sys.reward(states, actions, disturbances)

    def dynamics(self, states, actions, disturbances):
        actions = self.scale_action(actions, states)
        return self.sys.dynamics(states, actions, disturbances)

    def disturbance(self, states, actions):
        actions = self.scale_action(actions, states)
        return self.sys.disturbance(states, actions)


class MGNoGen(SystemWrapper):
    def __init__(self, sys):
        super(MGNoGen, self).__init__(sys)
        self.sys = sys
        self.action_space_size = 1
        self.act_min = self.sys.act_min.split(1, dim=0)[0]
        self.act_max = self.sys.act_max.split(1, dim=0)[0]

    def augment_action(self, action):
        p_bat = action
        p_gen = torch.zeros_like(p_bat)
        return torch.cat((p_bat, p_gen), dim=-1)

    def sample_action(self, state):
        action = torch.empty((state.shape[0], self.act_min.shape[0]), device=self.device).uniform_(self.act_min.item(),
                                                                                                   self.act_max.item())
        return action

    def reward(self, states, actions, disturbances):
        actions = self.augment_action(actions)
        return self.sys.reward(states, actions, disturbances)

    def dynamics(self, states, actions, disturbances):
        actions = self.augment_action(actions)
        return self.sys.dynamics(states, actions, disturbances)

    def disturbance(self, states, actions):
        actions = self.augment_action(actions)
        return self.sys.disturbance(states, actions)

    def initial_state(self, number_trajectories):
        """ samples 'number_trajectories' initial states from P_0
         returns a tensor of shape ('number_trajectories', |S|) """
        init_state = self.sys.initial_state(number_trajectories)
        self.env_min = self.sys.env_min
        self.env_max = self.sys.env_max
        self.act_min = self.sys.act_min.split(1, dim=0)[0]
        self.act_max = self.sys.act_max.split(1, dim=0)[0]
        return init_state
