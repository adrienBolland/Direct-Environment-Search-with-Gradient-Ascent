from optimizer.optimize import optimize

from multiprocessing import Pool
from functools import partial
import numpy as np

from systems import *
from policies import *
from config import *


def grid_optimization(nb_run, nb_processes, **kwargs):
    # get the parameters
    parameters = eval(kwargs["grid_parameters"])(**kwargs)

    # create pools
    with Pool(processes=nb_processes) as pool:
        # ensure systems fir is not set
        kwargs["systems"] = False

        results = pool.map(partial(search, *[nb_run], **kwargs),
                           parameters["list"])

    # create one grid per type of result
    grids = {key: np.copy(parameters['grid']) for key in results[0]}

    # NOTE: parameters["idx"] contains the parameters idx in the grid
    for idx, r in zip(parameters["idx"], results):
        for key, value in r.items():
            grids[key][idx] = value

    # Need to convert numpy array to list for serialization
    for key in grids:
        grids[key] = grids[key].tolist()  # not super efficient

    return {"xy": parameters["parameters"], "z": grids}


def search(nb_run, parameters, **kwargs):
    # Initialize the environment and policy
    system_nn = eval(kwargs["system_name"]).initialize(**kwargs["system_args"])
    for wrapper_name, values in kwargs["wrappers"].items():
        system_nn = eval(wrapper_name)(system_nn, **values)
    pol_nn = eval(kwargs["policy_name"]).initialize(**kwargs["policy_args"])

    # list of results for one single set of parameters
    perf_one_param = []
    dist_one_param = []

    # average the results over nb_run runs
    for _ in range(nb_run):
        # set the environment
        system_nn.unwrapped.set_parameters(**parameters)
        pol_nn.reset_parameters(**parameters)

        # fit
        _, agent = optimize(system_nn, pol_nn, None, **kwargs)
        perf, dist = agent.avg_performance(100)

        perf_one_param.append(perf)
        dist_one_param.append(dist)

    # provide the mean and the std of the results
    return {"Expected return": np.mean(perf_one_param),
            "return std": np.std(perf_one_param),
            "Expected equilibrium": np.mean(dist_one_param),
            "equilibrium std": np.std(dist_one_param)}
