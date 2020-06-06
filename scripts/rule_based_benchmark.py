from functools import partial

import numpy as np

import utils
from policies.Policy import NaiveMGPolicy, NaiveMGPolicyGenFirst
from systems.System import MicroGridOld, RewardScaling
import torch
from scipy.optimize import minimize

MG_CONFIG = {
    "config_file": "config/config_1.json",
    "experiment": "run",
}
args = utils.load_json(MG_CONFIG["config_file"])

system_args = args["system_args"]
horizon = system_args["horizon"]
num_trj = 2


def f(size, mg, rlb, opt=True):
    size = np.clip(size, 0., np.inf)
    mg.unwrapped.pv_size_init = size[0]
    mg.unwrapped.bat_size_init = size[1]
    mg.unwrapped.gen_size_init = size[2]
    # rlb.pv_size = size[0]
    # rlb.bat_size = size[1]
    # rlb.gen_size = size[2]
    rlb.reset_parameters(size[0], size[1])
    mg.reset_parameters()
    states = []
    actions = []
    rewards = []
    dist = []
    s = mg.initial_state(number_trajectories=num_trj)

    cum_rew = torch.zeros((num_trj, 1))
    for t in range(horizon):
        a = rlb(s)
        states.append(s)
        actions.append(a)
        next_states, disturbances, reward, action = mg.forward(s, a)
        rewards.append(reward)
        cum_rew += reward
        dist.append(disturbances)
        s = next_states.clone()
    cost = torch.mean(cum_rew).detach().item()  # * inv_years
    print(size, cost)
    if opt:
        return -cost
    else:
        return cost, states, actions, dist, rewards, num_trj


OPT_GLAG = False

if __name__ == '__main__':
    horizon = system_args["horizon"]
    mg = MicroGridOld(**system_args)
    mg = RewardScaling(mg)
    # mg = RewardExpScaling(mg)
    # size_0 = np.array([111.19692145,
    #                    58.96444597,
    #                    6.8020887])
    # -43304.8984375
    # size_1 = np.array([126.81768228,
    #                    84.08166826,
    #                    2.19175925])

    # size_0 = np.array([146.9,
    #                    146.6,
    #                    7.28])
    # size_1 = np.array([146.9,
    #                    146.6,
    #                    7.28])
    size_0 = np.array([system_args["pv_size"], system_args["bat_size"], system_args["gen_size"]])
    size_1 = np.array([system_args["pv_size"], system_args["bat_size"], system_args["gen_size"]])
    rlb1 = NaiveMGPolicy(2, 2, system_args["pv_size"], system_args["dem_size"], system_args["bat_size"],
                         system_args["gen_size"], system_args["charge_eff"], system_args["discharge_eff"])

    rlb2 = NaiveMGPolicyGenFirst(2, 2, system_args["pv_size"], system_args["dem_size"], system_args["bat_size"],
                                 system_args["gen_size"], system_args["charge_eff"], system_args["discharge_eff"])
    if not OPT_GLAG:
        c, states, actions, dist, rewards, num_trj = f(size_0, mg, rlb1, opt=OPT_GLAG)

        mg.render(
            torch.stack(states, dim=1).squeeze(dim=0),
            torch.stack(actions, dim=1).squeeze(dim=0),
            torch.stack(dist, dim=1).squeeze(dim=0),
            torch.stack(rewards, dim=1).squeeze(dim=0),
            num_trj)

        c, states, actions, dist, rewards, num_trj = f(size_1, mg, rlb2, opt=OPT_GLAG)
        states, actions, dist, rewards = torch.stack(states, dim=1).squeeze(dim=0),\
        torch.stack(actions, dim=1).squeeze(dim=0),\
        torch.stack(dist, dim=1).squeeze(dim=0),\
        torch.stack(rewards, dim=1).squeeze(dim=0)
        states, actions, rewards = utils.down_scale_trj(mg, states, actions, rewards)

        mg.render(states, actions, dist, rewards, num_trj)

    else:
        opt_fun_1 = partial(f, mg=mg, rlb=rlb1, opt=True)
        res_1 = minimize(opt_fun_1, size_0, method='nelder-mead',
                         options={'xatol': 1., 'fatol': 10., 'disp': True, "maxiter": 200})

        opt_fun_2 = partial(f, mg=mg, rlb=rlb2, opt=True)
        res_2 = minimize(opt_fun_2, size_1, method='nelder-mead',
                         options={'xatol': 1., 'fatol': 10., 'disp': True, "maxiter": 200})
