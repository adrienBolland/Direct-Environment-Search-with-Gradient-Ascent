import torch
from torch import nn


class System(nn.Module):
    """ rem : the object manipulates the actions as outputted by the policy network"""

    def __init__(self, device="cpu"):
        super(System, self).__init__()
        self.device = device

    def reward(self, states, actions, disturbances):
        """ reward function rho(s_t, a_t, xi_t) -> r_t"""
        raise NotImplementedError

    def dynamics(self, states, actions, disturbances):
        """ dynamics f(s_t, a_t, xi_t) -> s_t+1 """
        raise NotImplementedError

    def disturbance(self, states, actions):
        """ disturbance distribution P_xi(.|s_t, a_t)
         returns a torch.distribution object """
        raise NotImplementedError

    def render(self, states, actions, dist, rewards, num_trj):
        pass

    def forward(self, state, action):
        """Gathers the disturbance, dynamics and reward """
        raise NotImplementedError

    def initial_state(self, number_trajectories):
        """ samples 'number_trajectories' initial states from P_0
         returns a tensor of shape ('number_trajectories', |S|) """
        raise NotImplementedError

    def project_parameters(self):
        """ performs parameters projection """
        raise NotImplementedError

    def reset_parameters(self):
        """ reset the parameters (require gradient tensors) """
        raise NotImplementedError

    def parameters_dict(self):
        """ returns a dictionary mapping the parameters' name to their values """
        return dict()

    def stabilization_estimate(self, states_batch):
        """ From a batch of states series of shape (:, T, |S|) assess the performance of the controller """
        return torch.zeros(1, device=self.device)

    @staticmethod
    def initialize(**kwargs):
        """ instantiate an object from a dictionary """
        raise NotImplementedError

    @property
    def unwrapped(self):
        """Completely unwrap this systems
        """
        return self
