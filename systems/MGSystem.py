import torch
from torch import nn
from torch.distributions import Normal
import matplotlib.pyplot as plt

from systems.System import System


class MicroGrid(System):

    def __init__(self, dem_size, pv_size, bat_size, gen_size, power_rating, charge_eff, discharge_eff, bat_cost,
                 pv_cost, gen_cost, inv_rate, inv_years, horizon, fuel_price, load_curtail_price, load_shed_price,
                 device="cpu"):
        super(MicroGrid, self).__init__(device=device)
        self.dem_size = torch.tensor([dem_size], dtype=torch.float32, device=self.device)
        self.pv_size_init = torch.tensor([pv_size], dtype=torch.float32, device=self.device)
        self.bat_size_init = torch.tensor([bat_size], dtype=torch.float32, device=self.device)
        self.gen_size_init = torch.tensor([gen_size], dtype=torch.float32, device=self.device)

        self.power_rating = torch.tensor([power_rating], dtype=torch.float32, device=self.device)
        self.charge_eff = torch.tensor([charge_eff], dtype=torch.float32, device=self.device)
        self.discharge_eff = torch.tensor([discharge_eff], dtype=torch.float32, device=self.device)
        self.bat_cost = torch.tensor([bat_cost], dtype=torch.float32, device=self.device)
        self.pv_cost = torch.tensor([pv_cost], dtype=torch.float32, device=self.device)
        self.gen_cost = torch.tensor([gen_cost], dtype=torch.float32, device=self.device)
        self.inv_rate = torch.tensor([inv_rate], dtype=torch.float32, device=self.device)
        self.years = torch.tensor([inv_years], dtype=torch.float32, device=self.device)

        self.fuel_price = torch.tensor([fuel_price], dtype=torch.float32, device=self.device)
        self.load_shed_price = torch.tensor([load_shed_price], dtype=torch.float32, device=self.device)
        self.load_curtail_price = torch.tensor([load_curtail_price], dtype=torch.float32, device=self.device)
        self.pv_size = nn.Parameter(torch.Tensor([1.]), requires_grad=True)
        self.bat_size = nn.Parameter(torch.tensor([1.]), requires_grad=True)
        self.gen_size = nn.Parameter(torch.Tensor([1.]), requires_grad=False)
        self.horizon = torch.tensor([horizon], dtype=torch.float32, device=self.device)

        self.pv_avg_prod = torch.tensor([0.00000001e+00, 0.00000001e+00, 0.00000001e+00, 0.00000001e+00,
                                         0.00000001e+00, 0.00000001e+00, 0.00000001e+00, 0.00000001e+00,
                                         0.00000001e+00, 4.62232374e-02, 8.89720101e-02, 1.22127062e-01,
                                         1.41992336e-01, 1.49666484e-01, 1.43378674e-01, 1.20629623e-01,
                                         8.71089652e-02, 4.64848134e-02, 1.84307861e-17, 0.00000001e+00,
                                         0.00000001e+00, 0.00000001e+00, 0.00000001e+00, 0.00000001e+00],
                                        device=self.device)

        self.dem_avg = torch.tensor([0.3457438, 0.32335429, 0.309672, 0.29759948, 0.28587788,
                                     0.27293944, 0.24240862, 0.22680175, 0.23042503, 0.23326265,
                                     0.23884741, 0.24825482, 0.25547133, 0.26739509, 0.27287241,
                                     0.27219202, 0.2706911, 0.29403735, 0.42060912, 0.53479381,
                                     0.5502525, 0.5267475, 0.46403763, 0.39285948], device=self.device)

        self.dist_std = torch.tensor([0.05542831, 0.05022998, 0.0432726, 0.03978419, 0.03952021,
                                      0.03775034, 0.03728352, 0.03621157, 0.04035931, 0.04320152,
                                      0.04408169, 0.04740461, 0.04239965, 0.04087229, 0.04240869,
                                      0.04717433, 0.0436305, 0.04424234, 0.08158905, 0.06022856,
                                      0.0553013, 0.05767294, 0.06095378, 0.05918214], device=self.device)

        self.observation_space_size = 4
        self.action_space_size = 2

        self.reset_parameters()
        self.env_min = torch.tensor([0., 0., 0., 0.],
                                    device=self.device)
        self.env_max = torch.tensor([self.bat_size.item(), 23, self.pv_size.item(), self.dem_size.item()],
                                    device=self.device)
        self.act_min = torch.tensor([-self.power_rating * self.bat_size.item(), 0.], device=self.device)
        self.act_max = torch.tensor([self.power_rating * self.bat_size.item(), self.gen_size.item()],
                                    device=self.device)

    def reset_parameters(self):
        nn.init.constant_(self.pv_size, self.pv_size_init.item())
        nn.init.constant_(self.bat_size, self.bat_size_init.item())
        nn.init.constant_(self.gen_size, self.gen_size_init.item())

    def set_parameters(self, pv, bat):
        nn.init.constant_(self.pv_size, pv)
        nn.init.constant_(self.bat_size, bat)

    def initial_state(self, number_trajectories=1):

        self.env_min = torch.tensor([0., 0., 0., 0.],
                                    device=self.device)
        self.env_max = torch.tensor([self.bat_size.item(), 23, self.pv_size.item(), self.dem_size.item()],
                                    device=self.device)
        self.act_min = torch.tensor([-self.power_rating * self.bat_size.item(), 0.], device=self.device)
        self.act_max = torch.tensor([self.power_rating * self.bat_size.item(), self.gen_size.item()],
                                    device=self.device)

        soc = torch.empty((number_trajectories, 1), device=self.device).uniform_(100. * self.bat_size.item() / 2.,
                                                                                 100. * self.bat_size.item() / 2.)
        h = torch.empty((number_trajectories, 1), device=self.device).zero_()
        # soc = torch.empty((number_trajectories, 1), device=self.device).uniform_(100. * self.bat_size.item() / 2.,
        #                                                                          100. * self.bat_size.item())
        # h = torch.empty((number_trajectories, 1), device=self.device).uniform_(0., 23.)
        return self.construct_state(soc, h)

    def construct_state(self, soc, h):
        avg_pv, avg_dem = self._get_avg_pv_dem(h)
        state = torch.cat((soc, h, avg_pv, avg_dem), dim=1)
        return state

    def _get_avg_pv_dem(self, h):
        avg_pv = self.pv_avg_prod[h.long()].clone().detach().reshape((-1, 1)) * self.pv_size * 100.
        avg_dem = self.dem_avg[h.long()].clone().detach().reshape((-1, 1)) * self.dem_size
        return avg_pv, avg_dem

    def disturbance(self, state, action):

        h = state[:, 1]

        mu = torch.zeros(state.shape[0], device=self.device).reshape((-1, 1))
        sigma = self.dist_std[h.long()].reshape((-1, 1))

        return Normal(mu, sigma)

    def reward(self, state, action, disturbances):
        action = self.check_actions(action)
        soc, h = state[:, 0].reshape(-1, 1), state[:, 1].reshape(-1, 1)
        avg_pv, avg_dem = self._get_avg_pv_dem(h)
        c_t, pv_t = avg_dem + disturbances, avg_pv
        # c_t, pv_t = avg_dem, avg_pv
        p_bat, p_gen = action.split(1, dim=1)

        curt = torch.clamp(-p_bat - (self.bat_size * 100. - soc) / self.charge_eff, min=0)
        ch = torch.clamp(-p_bat - curt, min=0)

        res_load = torch.clamp(p_bat - soc * self.discharge_eff, min=0.)
        dis = torch.clamp(p_bat - res_load, min=0.)
        p_bat = dis - ch
        diff = p_gen + pv_t + p_bat - c_t
        gen_cost = p_gen * self.fuel_price
        inv_cost = self.amortization()
        load_shed_cost = torch.clamp(-diff * self.load_shed_price, min=0)
        curt_cost = torch.clamp(diff * self.load_curtail_price, min=0)  # + (res_load + curt) * self.load_curtail_price
        reward = -(inv_cost + gen_cost + load_shed_cost + curt_cost) * 8760. / float(self.horizon)
        # reward = -(gen_cost + load_shed_cost + curt_cost) * 8760 / float(self.horizon)
        return reward.view(-1, 1)

    def forward(self, state, action):

        disturbances = self.disturbance(state, action).sample()

        reward = self.reward(state, action, disturbances)

        next_states = self.dynamics(state, action, disturbances)
        return next_states, disturbances, reward, action

    def dynamics(self, state, action, disturbances):
        action = self.check_actions(action)
        soc, h = state[:, 0].reshape(-1, 1), state[:, 1].reshape(-1, 1)
        p_bat, p_gen = action.split(1, dim=1)
        n_s = self.battery_dynamics(soc, p_bat)
        next_soc = self.check_bat_cap_limits(n_s)
        next_h = (h + 1) % 24

        return self.construct_state(next_soc.reshape(-1, 1),
                                    next_h.reshape(-1, 1))

    def battery_dynamics(self, soc, p_bat):
        """
        p_bat>0 : discharge
        p_bat<0 : charge
        """

        n_s = soc - torch.clamp(p_bat, max=0) * self.charge_eff - torch.clamp(p_bat, min=0) / self.discharge_eff
        return n_s

    def check_actions(self, action):
        p_bat, p_gen = action.split(1, dim=1)

        p_bat = self.check_bat_limits(p_bat)
        p_gen = self.check_gen_limits(p_gen)
        action = torch.cat((p_bat.reshape(-1, 1), p_gen.reshape(-1, 1)), dim=1)
        return action

    def check_bat_limits(self, p_bat):
        p_bat = self.clamp_max(p_bat, self.power_rating * self.bat_size * 100.)
        p_bat = self.clamp_min(p_bat, -self.power_rating * self.bat_size * 100.)
        return p_bat

    def check_gen_limits(self, p_gen):
        p_gen = torch.clamp(p_gen, min=0.)
        p_gen = self.clamp_max(p_gen, self.gen_size)
        return p_gen

    def check_bat_cap_limits(self, soc):
        n_soc = torch.clamp(soc, min=0.)
        n_soc = self.clamp_max(n_soc, self.bat_size * 100.)
        return n_soc

    def amortization(self):
        inv_pv = self.pv_size * 100. * self.pv_cost
        inv_bat = self.bat_size * 100. * self.bat_cost
        inv_gen = self.gen_size * self.gen_cost
        tot_inv = inv_bat + inv_gen + inv_pv
        amort = tot_inv * (self.inv_rate * (1 + self.inv_rate) ** self.years) / (
                (1 + self.inv_rate) ** self.years - 1) / 8760.
        return amort

    def project_parameters(self):
        with torch.no_grad():
            pv_ = torch.clamp(self.pv_size, min=0.)
            bat_ = torch.clamp(self.bat_size, min=0.)
            gen_ = torch.clamp(self.gen_size, min=0.)
        nn.init.constant_(self.pv_size, pv_.item())
        nn.init.constant_(self.bat_size, bat_.item())
        nn.init.constant_(self.gen_size, gen_.item())

    def sample_action(self, state):
        lim_bat = self.power_rating.item() * self.bat_size.item() * 100.
        b_p = torch.empty((state.shape[0], 1), device=self.device).uniform_(-lim_bat, lim_bat)

        lim_gen = self.gen_size.item()
        g_p = torch.empty((state.shape[0], 1), device=self.device).uniform_(0., lim_gen + 50)
        return torch.cat((b_p, g_p), dim=1)

    def render(self, states, actions, dist, rewards, num_trj):
        p_bat, p_gen = actions.split(1, dim=2)
        p_bat = self.check_bat_limits(p_bat)
        p_gen = self.check_gen_limits(p_gen)
        actions = torch.cat((p_bat, p_gen), dim=2)
        s = states.to('cpu').detach().numpy()
        a = actions.to('cpu').detach().numpy()
        d = dist.to('cpu').detach().numpy()
        r = rewards.detach().to('cpu').detach().numpy()
        for i in range(num_trj):
            f, axs = plt.subplots(4, 1, sharex=True)
            for ax, y, t in zip(axs, [s[i], a[i], d[i], r[i]], ['States', "Actions", "Disturbances", "Rewards"]):
                ax.set_title(t)
                ax.plot(y)

    @staticmethod
    def initialize(**kwargs):
        return MicroGrid(**kwargs)

    def stabilization_estimate(self, states_batch):
        return torch.zeros(1, device=self.device)

    def parameters_dict(self):
        return {'pv_size': self.pv_size.item(),
                'bat_size': self.bat_size.item(),
                'gen_size': self.gen_size.item()}

    @staticmethod
    def clamp_min(x, val):
        """
         Clamps x to minimum value 'val'.
         val < 0.0
        """
        return x.clamp(max=0.0).sub(val).clamp(min=0.0).add(val) + x.clamp(min=0.0)

    @staticmethod
    def clamp_max(x, val):
        """
         Clamps x to maximum value 'val'.
         val > 0.0
         """
        return x.clamp(min=0.0).sub(val).clamp(max=0.0).add(val) + x.clamp(max=0.0)
