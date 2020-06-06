from torch import nn


class Policy(nn.Module):

    def __init__(self, device="cpu"):
        super(Policy, self).__init__()
        self.device = device

    def forward(self, x):
        raise NotImplementedError()

    def distribution(self, output_net):
        class Dist:
            def __init__(self, actions):
                self.actions = actions

            def sample(self):
                return self.actions

            def log_prob(self, actions):
                raise Exception("Unimplemented log likelihood for deterministic distributions")

        return Dist(output_net)

    def reset_parameters(self, **kwargs):
        raise NotImplementedError()

    def project_parameters(self):
        # unconstrained parameters
        pass
