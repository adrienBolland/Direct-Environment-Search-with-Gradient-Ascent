import torch
from torch import nn
from torch.distributions import MultivariateNormal

from policies.MLPPolicy import MLPPolicy


class MGStationaryPolicy(nn.Module):

    def __init__(self, input_size, n_actions, layers, act_fun=nn.Tanh(), scale=None,
                 normalize=None, reg_layer=None, device="cpu"):
        super(MGStationaryPolicy, self).__init__()
        self.device = device
        self.output_size = n_actions

        init_mlp = {"input_size": input_size,
                    "layers": layers,
                    "act_fun": act_fun,
                    "n_output": 2 * self.output_size,
                    "scale": scale,
                    "normalize": normalize,
                    "reg_layer": reg_layer,
                    "device": self.device}

        self.n = MLPPolicy(**init_mlp)
        self._cov = nn.Parameter(torch.tensor((1.)), requires_grad=False)
        self._mean = nn.Parameter(torch.tensor((1.)), requires_grad=False)

    def forward(self, x):
        return self.n(x)

    def distribution(self, output_net):
        def distribution(x):
            mean = x[:, :self.output_size]

            cov = torch.diag_embed(x[:, self.output_size:].exp() + 1e-04)
            self._mean = nn.Parameter(mean.clone().detach(), requires_grad=False)
            self._cov = nn.Parameter(cov.clone().detach(), requires_grad=False)
            return MultivariateNormal(mean, cov)

        return distribution(output_net)

    def reset_parameters(self, **kwargs):
        self.n.reset_parameters()

    def project_parameters(self):
        # unconstrained parameters
        pass

    @staticmethod
    def initialize(**kwargs):
        return MGStationaryPolicy(**kwargs)
