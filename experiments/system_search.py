import os
from copy import deepcopy
from multiprocessing import Pool
from functools import partial

from optimizer.optimize import optimize
from systems import *
from policies import *

from experiments.utils import DEVICE
import utils


def direct_search(log_path, **kwargs):
    # log writer
    writer = utils.LogWriter(log_path) if log_path is not None else None

    system_nn = eval(kwargs["system_name"]).initialize(**kwargs["system_args"], device=DEVICE)
    for wrapper_name, values in kwargs["wrappers"].items():
        system_nn = eval(wrapper_name)(system_nn, **values)
    kwargs["input_size"] = system_nn.observation_space_size
    kwargs["n_actions"] = system_nn.action_space_size
    pol_nn = eval(kwargs["policy_name"]).initialize(**kwargs["policy_args"], device=DEVICE)
    system_nn.to(DEVICE)
    pol_nn.to(DEVICE)

    # Fit environment
    env, agent = optimize(system_nn, pol_nn, writer, **kwargs)

    # save agent
    utils.save_agent(log_path, agent)

    agent.evaluate_performance(4)
    return agent


def multiple_direct_search(log_path, nb_run, nb_workers, **kwargs):
    # arguments for the runs
    seed = kwargs.get("seed", 42)

    argument_list = []
    for run_id in range(nb_run):
        # change the seed
        a = deepcopy(kwargs)
        a["seed"] = seed + run_id

        # change the path directory
        d = os.path.join(log_path, f"run-r{run_id}")

        # argument list
        argument_list.append((d, a))

    # create pools
    with Pool(processes=nb_workers) as pool:
        agent_list = pool.map(partial(_direct_arg), argument_list)

    return agent_list


def _direct_arg(pair):
    return direct_search(pair[0], **pair[1])
