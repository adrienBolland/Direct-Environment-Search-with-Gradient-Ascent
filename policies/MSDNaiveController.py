import torch

from policies.Policy import Policy
from policies.utils import DEVICE


class NaiveController(Policy):
    def __init__(self, n_actions, equilibrium):
        super(NaiveController, self).__init__()

        self.n_actions = n_actions
        self.equilibrium = equilibrium

    def forward(self, x):
        action = torch.zeros((x.shape[0], self.n_actions), device=DEVICE)
        action[x[:, 0] >= self.equilibrium, 0] = 1.
        action[x[:, 0] < self.equilibrium, -1] = 1.

        return action

    def distribution(self, output_net):
        return super(NaiveController, self).distribution(output_net)

    def reset_parameters(self, **kwargs):
        pass

    @staticmethod
    def initialize(**kwargs):
        return NaiveController(**kwargs)
