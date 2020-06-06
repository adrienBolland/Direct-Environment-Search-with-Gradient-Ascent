import torch

from policies.Policy import Policy


class NaiveMGPolicy(Policy):

    def __init__(self, n_actions, input_size, pv_size, dem_size, bat_size, gen_size, charge_eff, discharge_eff):
        super(NaiveMGPolicy, self).__init__()
        # output size
        self.output_size = n_actions

        # input size
        self.input_size = input_size - 1
        self.pv_size = pv_size * 100.
        self.dem_size = dem_size
        self.bat_size = bat_size * 100.
        self.gen_size = gen_size
        self.charge_eff = charge_eff
        self.discharge_eff = discharge_eff
        # Common Neural Net #

    def forward(self, state):
        soc, h, pv, dem = state[:, 0], state[:, 1], state[:, 2], state[:, 3]

        diff = pv - dem

        curt = torch.clamp(diff - (self.bat_size - soc) / self.charge_eff, min=0)
        ch = torch.clamp(diff - curt, min=0)

        res_load = torch.clamp(-diff - soc * self.discharge_eff, min=0.)
        dis = torch.clamp(-diff - res_load, min=0.)
        gen = torch.clamp(res_load, min=0., max=self.gen_size)

        return torch.cat(((dis - ch).reshape(-1, 1), gen.reshape(-1, 1)), dim=1)

    def distribution(self, output_net):
        super(NaiveMGPolicy, self).distribution(output_net)

    def reset_parameters(self, pv, bat):
        self.pv_size = pv * 100.
        self.bat_size = bat * 100.

    @staticmethod
    def initialize(**kwargs):
        return NaiveMGPolicy(input_size=kwargs["input_size"], n_actions=kwargs["n_actions"],
                             pv_size=kwargs["system_args"]["pv_size"], dem_size=kwargs["system_args"]["dem_size"],
                             bat_size=kwargs["system_args"]["bat_size"], gen_size=kwargs["system_args"]["gen_size"],
                             charge_eff=kwargs["system_args"]["charge_eff"],
                             discharge_eff=kwargs["system_args"]["discharge_eff"])


class NaiveMGPolicyGenFirst(NaiveMGPolicy):
    def __init__(self, n_actions, input_size, pv_size, dem_size, bat_size, gen_size, charge_eff, discharge_eff):
        super(NaiveMGPolicyGenFirst, self).__init__(n_actions, input_size, pv_size, dem_size, bat_size, gen_size,
                                                    charge_eff, discharge_eff)

    def forward(self, state):
        soc, h, pv, dem = state[:, 0], state[:, 1], state[:, 2], state[:, 3]

        diff = pv - dem

        curt = torch.clamp(diff - (self.bat_size - soc) / self.charge_eff, min=0)
        ch = torch.clamp(diff - curt, min=0)

        gen = torch.clamp(-diff, min=0., max=self.gen_size)
        res_load = torch.clamp(-diff - gen, min=0.)
        load_shed = torch.clamp(res_load - soc * self.discharge_eff, min=0.)
        dis = torch.clamp(res_load - load_shed, min=0.)
        return torch.cat(((dis - ch).reshape(-1, 1), gen.reshape(-1, 1)), dim=1)
