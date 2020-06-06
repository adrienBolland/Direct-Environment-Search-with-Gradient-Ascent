import torch
from torch import nn
from torch.distributions import OneHotCategorical

from policies.Policy import Policy


class MLPPolicy(Policy):
    def __init__(self, input_size, n_output, layers=(10,), act_fun=nn.Tanh(), scale=None,
                 normalize=None, reg_layer=None, device="cpu"):
        super(MLPPolicy, self).__init__()
        self.device = device
        self.num_actions = n_output

        if scale is None:
            self.scale = torch.zeros((1, input_size), device=self.device)
        else:
            self.scale = torch.tensor([scale], device=self.device)

        if normalize is None:
            self.normalize = torch.ones((1, input_size), device=self.device)
        else:
            self.normalize = torch.tensor([normalize], device=self.device)

        self.layers = []
        for n_neurons in layers:
            # linear layers
            self.layers.append(nn.Linear(input_size, n_neurons))
            self.layers.append(act_fun.__class__())

            # add regularization layer if required
            if reg_layer is not None:
                self.layers.append(reg_layer.__class__())

            input_size = n_neurons

        self.layers.append(nn.Linear(input_size, n_output))

        self.net = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.net((x - self.scale) / self.normalize)

    def distribution(self, output_net):
        raise NotImplementedError()

    def reset_parameters(self, **kwargs):

        def weight_reset(m):
            if isinstance(m, nn.Linear):
                m.reset_parameters()

        self.net.apply(weight_reset)

    def project_parameters(self):
        # unconstrained parameters
        pass

    @staticmethod
    def initialize(**kwargs):
        return MLPPolicy(**kwargs)


class MLPCategoricalPolicy(MLPPolicy):
    def __init__(self, **kwargs):
        super(MLPCategoricalPolicy, self).__init__(**kwargs)

    def forward(self, x):
        return super(MLPCategoricalPolicy, self).forward(x)

    def distribution(self, output_net):
        return OneHotCategorical(logits=output_net)

    @staticmethod
    def initialize(**kwargs):
        return MLPCategoricalPolicy(**kwargs)
