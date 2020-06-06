import json
import pickle
import os
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
import random

from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
from matplotlib.ticker import FixedLocator


def get_version(name="direct-mdp", logdir='logs/', width=3):
    """returns str repr of the version"""
    os.makedirs(logdir, exist_ok=True)
    files = list(sorted([f for f in os.listdir(logdir) if f"{name}-v" in f]))
    if len(files) < 1:
        version = '1'.rjust(width, '0')
    else:
        last_version = int(files[-1][-width:])
        version = str(last_version + 1).rjust(width, '0')
    return version


def list_run_dirs(logdir):
    run_dirs = [os.path.join(logdir, f) for f in os.listdir(logdir) if "run-r" in f]
    return run_dirs


def load_json(path):
    with open(path, "r") as json_f:
        config_dict = json.load(json_f)

    return config_dict


def save_config(path, config_dict):
    """drops config dictionary inside json file"""
    path = os.path.join(path, "config.json")
    return save_json(path, config_dict)


def save_json(path, dic):
    dir, file = os.path.split(path)
    if dir != '':  # current
        os.makedirs(dir, exist_ok=True)  # required if directory not created yet

    with open(path, "w") as json_f:
        json.dump(dic, json_f)


def save_agent(dirpath, agent):
    with open(os.path.join(dirpath, "model.pickle"), "wb") as pickle_f:
        pickle.dump(agent, pickle_f)


def load_agent(dirpath):
    with open(os.path.join(dirpath, "model.pickle"), "rb") as pickle_f:
        agent = pickle.load(pickle_f)
    return agent


def set_seeds(seed_value):
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)


class LogWriter:
    """
    simple wrapper encapsulating tensorboard logging operations
    """

    def __init__(self, log_path):
        self.writer = SummaryWriter(log_path)

    def add_return(self, value, step):
        prefix = "return/"
        self.writer.add_scalar(f"{prefix}return", value, step)

    def add_loss(self, loss, step):
        prefix = "loss/"
        for n, l in loss.items():
            self.writer.add_scalar(f"{prefix}{n}", l, step)

    def add_expected_return(self, perf, step):
        prefix = "performance/"  # name for backward compatibility
        self.writer.add_scalar(f"{prefix}agent", perf, step)

    def add_grad_histograms(self, params_dict, step):
        for net, params in params_dict.items():
            prefix = f"{net}/"
            for name, param in params:
                if param.grad is not None:
                    self.writer.add_histogram(f"{prefix}grad-{name}", param.grad.data, step)
                else:
                    self.writer.add_histogram(f"{prefix}grad-{name}", param.data, step)

    def add_policy_histograms(self, actions, step):
        """
        plots distribution of the parameters of the policy' distribution.
        e.g. in MSD, will plot distribution of logits of the categorical distribution over the actions,
            in MG, will plot the distribution of the parameters of the multi-dimensional gaussian.
        """
        prefix = "policy/"
        for p in range(actions.shape[1]):
            self.writer.add_histogram(f"{prefix}output-{p}", actions[:, p], step)

    def add_system_parameters(self, parameters_dict, step):
        prefix = "systems-params/"
        for name, param in parameters_dict.items():
            self.writer.add_scalar(f"{prefix}{name}", param, step)

    def add_disturbance_parameters(self, parameters_dict, step):
        prefix = "disturbance-params/"
        for name, param in parameters_dict.items():
            self.writer.add_scalar(f"{prefix}{name}", param, step)


def down_scale_trj(system, states, actions, rewards):
    identify = {"down_scale_state": None, "down_scale_action": None, "down_scale_reward": None, "augment_action": None}

    s = system
    for _ in range(100):
        for k in identify.keys():
            if hasattr(s, k):
                identify[k] = s
        if hasattr(s, "sys"):
            s = s.sys
        else:
            break
    states = identify["down_scale_state"].down_scale_state(states) if identify[
                                                                          "down_scale_state"] is not None else states
    actions = identify["augment_action"].augment_action(actions) if identify[
                                                                        "augment_action"] is not None else actions
    actions = identify["down_scale_action"].down_scale_action(actions) if identify[
                                                                              "down_scale_action"] is not None else actions
    rewards = identify["down_scale_reward"].down_scale_reward(rewards) if identify[
                                                                              "down_scale_reward"] is not None else rewards
    return states, actions, rewards


class CustomScale(mscale.ScaleBase):
    name = 'custom'

    def __init__(self, axis, *, limit=150, **kwargs):
        super().__init__(axis)
        self.limit = limit

    def get_transform(self):
        return self.CustomTransform()

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(FixedLocator([-1000, -500, 0, 50, 100, 150]))

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return vmin, self.limit

    class CustomTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
        has_inverse = True

        def __init__(self):
            mtransforms.Transform.__init__(self)

        def transform_non_affine(self, a):
            return a * (1 + (np.sign(a) + 1) * 5)

        def inverted(self):
            return CustomScale.InvertedCustomTransform()

    class InvertedCustomTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
        has_inverse = True

        def __init__(self):
            mtransforms.Transform.__init__(self)

        def transform_non_affine(self, a):
            return a / (1 + (np.sign(a) + 1) * 5)

        def inverted(self):
            return CustomScale.CustomTransform()
