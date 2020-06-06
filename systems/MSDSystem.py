import torch
from torch import nn
from torch.distributions import Normal

from systems.System import System


class MSDSystem(System):
    def __init__(self, equilibrium, actions_value, target_parameters_reward, cost_omega_zeta, accuracy, omega_interval,
                 zeta_interval, position_interval, speed_interval, phi_interval, actions_discretization, device="cpu"):
        super(MSDSystem, self).__init__(device=device)
        # equilibrium position
        self.equilibrium = equilibrium

        # action values
        self.actions_value = torch.tensor([actions_value], device=self.device).T

        # target constant parameters in the reward
        self.target_parameters = torch.tensor([target_parameters_reward], device=self.device).T
        s0, s1 = self.target_parameters.shape
        self.phi_param = nn.Parameter(torch.Tensor(s0, s1), requires_grad=True)

        # accuracy in the error (lambda)
        self.accuracy = accuracy

        # actions time discretization
        self.actions_discretization = actions_discretization

        # initial position and speed intervals
        self.position_interval = position_interval
        self.speed_interval = speed_interval

        # omega zeta parameters
        self.cost_omega_zeta = torch.tensor([cost_omega_zeta], device=self.device).T
        self.omega_interval = omega_interval
        self.zeta_interval = zeta_interval
        self.phi_interval = phi_interval

        self.omega_zeta_param = nn.Parameter(torch.Tensor(2, 1), requires_grad=True)

        # reset the parameters
        self.observation_space_size = 2
        self.action_space_size = len(self.actions_value)

        self.reset_parameters()
        self.env_min = torch.tensor([self.omega_interval[0], self.zeta_interval[0]],
                                    device=self.device)
        self.env_max = torch.tensor([self.omega_interval[1], self.zeta_interval[1]],
                                    device=self.device)
        self.act_min = torch.tensor([min(self.actions_value)], device=self.device)
        self.act_max = torch.tensor([max(self.actions_value)], device=self.device)

    def reset_parameters(self):
        nn.init.normal_(self.phi_param)
        nn.init.uniform_(self.omega_zeta_param[0, 0], self.omega_interval[0], self.omega_interval[1])
        nn.init.uniform_(self.omega_zeta_param[1, 0], self.zeta_interval[0], self.zeta_interval[1])

        return self

    def set_parameters(self, omega, zeta, constant_param):
        self.phi_param = nn.Parameter(torch.tensor([constant_param]).T, requires_grad=True)
        self.omega_zeta_param = nn.Parameter(torch.tensor([[omega, zeta]]).T, requires_grad=True)

    def project_parameters(self):
        with torch.no_grad():
            omega = self.omega_zeta_param[0, 0].clamp(self.omega_interval[0], self.omega_interval[1]).item()
            zeta = self.omega_zeta_param[1, 0].clamp(self.zeta_interval[0], self.zeta_interval[1]).item()
            phi = self.phi_param[:, 0].clamp(self.phi_interval[0], self.phi_interval[1]).tolist()

        nn.init.constant_(self.omega_zeta_param[0, 0], omega)
        nn.init.constant_(self.omega_zeta_param[1, 0], zeta)

        for i, p in enumerate(phi):
            nn.init.constant_(self.phi_param[i, 0], p)

    def dynamics(self, previous_states, one_hot_actions, disturbances):
        return MSDSystem.f(previous_states[:, 0:1],
                           previous_states[:, 1:2],
                           torch.matmul(one_hot_actions, self.actions_value) + disturbances,
                           self.actions_discretization,
                           self.omega_zeta_param[0, 0],
                           self.omega_zeta_param[1, 0])

    @staticmethod
    def f(x_0, v_0, a, t, omega, zeta):
        omega_t = omega * t
        action = a / omega.pow(2)

        if zeta > 1.:
            root = (zeta.pow(2) - 1).sqrt()

            position = ((-zeta * omega_t).exp()
                        * ((x_0 - action) * (root * omega_t).cosh()
                           + (v_0 / omega + zeta * (x_0 - action)) / root * (root * omega_t).sinh())
                        + action)

            speed = ((-zeta * omega * (-zeta * omega_t).exp()
                      * ((x_0 - action) * (root * omega_t).cosh()
                         + (v_0 / omega + zeta * (x_0 - action)) / root * (root * omega_t).sinh()))
                     + ((-zeta * omega_t).exp()
                        * ((x_0 - action) * root * omega * (root * omega_t).sinh()
                           + (v_0 + zeta * (x_0 - action) * omega) * (root * omega_t).cosh())))

        elif zeta == 1.:

            position = ((x_0 - action) + (v_0 + omega * (x_0 - action)) * t) * (- omega_t).exp() + action

            speed = (((x_0 - action) + (v_0 + omega * (x_0 - action)) * t) * (-omega) * (- omega_t).exp()
                     + (v_0 + omega * (x_0 - action)) * (- omega_t).exp())

        else:
            root = (1 - zeta.pow(2)).sqrt()

            position = ((-zeta * omega_t).exp()
                        * ((x_0 - action) * (root * omega_t).cos()
                           + (v_0 / omega + zeta * (x_0 - action)) / root * (root * omega_t).sin())
                        + action)

            speed = ((-zeta * omega * (-zeta * omega_t).exp()
                      * ((x_0 - action) * (root * omega_t).cos()
                         + (v_0 / omega + zeta * (x_0 - action)) / root * (root * omega_t).sin()))
                     + ((-zeta * omega_t).exp()
                        * (- (x_0 - action) * root * omega * (root * omega_t).sin()
                           + (v_0 + zeta * (x_0 - action) * omega) * (root * omega_t).cos())))

        return torch.cat((position, speed), 1)

    def initial_state(self, number_trajectories=1):
        p = torch.empty((number_trajectories, 1), device=self.device).uniform_(self.position_interval[0],
                                                                               self.position_interval[1])
        s = torch.empty((number_trajectories, 1), device=self.device).uniform_(self.speed_interval[0],
                                                                               self.speed_interval[1])
        return torch.cat((p, s), dim=1)

    def reward(self, states, one_hot_actions, disturbances=None):
        position_error = torch.abs(states[:, :1] - self.equilibrium)
        parameters_error = torch.prod((self.phi_param - self.target_parameters).pow(2))
        omega_zeta_error = torch.sum((self.omega_zeta_param - self.cost_omega_zeta).pow(2))

        acc = torch.ones(position_error.shape, device=self.device) * self.accuracy

        error = (position_error / acc + omega_zeta_error + parameters_error)

        error = torch.exp(- error)

        return error

    def disturbance(self, states, one_hot_actions):
        actions = torch.matmul(one_hot_actions, self.actions_value)
        positions, speeds = states[:, 0:1], states[:, 1:2]

        mu = positions
        sigma = 0.1 * actions.abs() + speeds.abs() + 10 ** -6

        return Normal(mu, sigma)

    def forward(self, previous_states, one_hot_actions, disturbances=None):
        # action value
        actions = torch.matmul(one_hot_actions, self.actions_value)

        # reward first
        reward = self.reward(previous_states, one_hot_actions)

        # dynamics update second
        if disturbances is None:
            disturbances = self.disturbance(previous_states, one_hot_actions).sample()

        next_states = self.dynamics(previous_states, one_hot_actions, disturbances)

        return next_states, disturbances, reward, actions

    def stabilization_estimate(self, states_batch):
        return torch.mean(torch.abs(states_batch[:, :, 0] - self.equilibrium)).item()

    def parameters_dict(self):
        return {'omega': self.omega_zeta_param[0, 0].tolist(),
                'zeta': self.omega_zeta_param[1, 0].tolist(),
                'phi-0': self.phi_param[0, 0].tolist(),
                'phi-1': self.phi_param[1, 0].tolist(),
                'phi-2': self.phi_param[2, 0].tolist()}

    @staticmethod
    def initialize(**kwargs):
        return MSDSystem(**kwargs)

    @staticmethod
    def states_repr():
        return {0: {"title": '$x_{t}$ - mass position [m]',
                    "name": 'position'},
                1: {"title": '$s_{t}$ - mass speed [m/s]',
                    "name": 'speed'}}
