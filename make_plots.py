import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.ndimage import uniform_filter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

plt.style.use('classic')

import utils

from matplotlib import scale as mscale

mscale.register_scale(utils.CustomScale)

# Rendering options for the plots
PLOT_CONTEXT = {
    'font.family': 'serif',
    'font.serif': 'Computer Modern Sans',
    'text.usetex': True,
    'font.size': 28,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'axes.formatter.useoffset': False
}

# Plots that can be made from tensorboard logs for each systems
# NOTE: 'tb_fields' is a dict 'tensorboard scalar' -> 'legend'
# NOTE: first-level key is the systems name
TB_PLOTS = {"MSDSystem":
                {'expected_return': {"xlabel": "Iteration $k$",
                                     "ylabel": "Expected return $V(\\psi_k, \\theta_k)$",
                                     "tb_fields": {'performance/agent': 'expected return'}},
                    'return': {"xlabel": "Iteration $k$",
                               "ylabel": "return",
                               "tb_fields": {'return/return': 'return'}},
                    'parameters': {"xlabel": "Iteration $k$",
                                   "ylabel": "parameters",
                                   "tb_fields": {'systems-params/omega': '$\omega$',
                                                 'systems-params/zeta': '$\zeta$',
                                                 'systems-params/phi-0': '$\phi_{0}$',
                                                 'systems-params/phi-1': '$\phi_{1}$',
                                                 'systems-params/phi-2': '$\phi_{2}$', }}
                 },
            "MicroGrid":
                {'expected_return': {"xlabel": "Iteration $k$",
                                     "ylabel": "Expected return $V(\\psi_k, \\theta_k)$",
                                     "tb_fields": {'performance/agent': 'expected return'}},
                   'return': {"xlabel": "Iteration $k$",
                              "ylabel": "return",
                              "tb_fields": {'return/return': 'return'}},
                 }
            }


def plot_trajectory(trajectory, states_repr, path, time_delta=.05, plot_context=None):
    if plot_context is None:
        plot_context = PLOT_CONTEXT
    with mpl.rc_context(plot_context):
        t = np.arange(len(trajectory))

        # States
        for s_id, repr in states_repr.items():
            s_ = list(map(lambda traj: traj[0][s_id], trajectory))

            f = plt.figure()
            plt.xlabel('$t$ - time [s]')
            plt.ylabel(repr['title'])
            plt.plot(t * time_delta, s_)
            plt.tight_layout()
            plt.ylim(min(s_) - .1 * abs((s_)),
                     max(s_) + .1 * abs(max(s_)))

            f.savefig(f"{path}/trajectory-state-{repr['name']}.pdf")
            plt.close(f)

        # actions
        actions = np.array(list(map(lambda traj: traj[1], trajectory)))
        if len(actions.shape) == 1:
            actions = actions.reshape(-1, 1)

        for a in range(actions.shape[1]):
            f = plt.figure()
            plt.xlabel('$t$ - time [s]')
            plt.ylabel('$a_{t}$ - action')
            plt.plot(t * time_delta, actions[:, a])
            plt.tight_layout()
            ymin, ymax = actions[:, a].min(), actions[:, a].max()
            plt.ylim(ymin - .05 * abs(ymax - ymin), ymax + .05 * abs(ymax - ymin))
            f.savefig(f"{path}/trajectory-action-{a}.pdf")
            plt.close(f)

        # reward
        rew_list = list(map(lambda traj: traj[3], trajectory))
        f = plt.figure()
        plt.xlabel('$t$ - time [s]')
        plt.ylabel(r'$\rho_{t}$ - reward')
        plt.plot(t * time_delta, rew_list)
        plt.tight_layout()
        ymin, ymax = min(rew_list), max(rew_list)
        plt.ylim(ymin - .05 * abs(ymax - ymin), ymax + .05 * abs(ymax - ymin))

        f.savefig(f"{path}/trajectory-reward.pdf")
        plt.close(f)


def plot_tb_logs(plots, event_accs, plots_path, plot_context=None):
    if plot_context is None:
        plot_context = PLOT_CONTEXT

    with mpl.rc_context(plot_context):
        for pname, pdict in plots.items():

            # get the fields in the event file
            f = plt.figure()
            plt.xlabel(pdict['xlabel'])
            plt.ylabel(pdict['ylabel'])
            plt.yscale(pdict.get('yscale', 'linear'))

            steps, vals = [], []

            for field in pdict["tb_fields"]:
                for event_acc in event_accs:
                    _, step, val = zip(*event_acc.Scalars(field))
                    steps.append(step), vals.append(list(val))  # NOTE: steps should be the same

                lens = []
                for i in vals:
                    lens.append(len(i))
                min_len = min(lens)
                steps = [y[:min_len] for y in steps]
                vals = [y[:min_len] for y in vals]

                if len(steps) == 1 and len(vals) == 1:  # Not mean / std
                    plt.plot(steps[0], vals[0])
                    ymin, ymax = min(vals[0]), max(vals[0])
                else:
                    mean_vals = np.array(vals).mean(axis=0)
                    std_vals = np.array(vals).std(axis=0)
                    l = plt.plot(steps[0], mean_vals)
                    color = l[-1].get_color()
                    y0, y1 = mean_vals - std_vals, mean_vals + std_vals
                    plt.fill_between(steps[0], y0, y1, alpha=.2, color=color)
                    ymin, ymax = min(y0), max(y1)

                plt.ylim(ymin - .05 * abs(ymax - ymin), ymax + .05 * abs(ymax - ymin))
                # plt.ylim(0, 120)

            if len(pdict["tb_fields"]) > 1:  # legend only if multiple lines
                labels = list(pdict["tb_fields"].values())
                plt.legend(labels)

            plt.tight_layout()
            f.savefig(f"{plots_path}/{pname}.pdf")
            plt.close(f)


def plot_grids(grids_dict, parameters, path, plot_context=None):
    assert "xy" in grids_dict and "z" in grids_dict
    assert len(parameters) == 2

    if plot_context is None:
        plot_context = PLOT_CONTEXT

    # retrieve desired parameters from results
    xname = parameters[0]
    x = grids_dict["xy"][xname]["values"]
    xlabel = grids_dict["xy"][xname]["name"]
    xidx = list(grids_dict["xy"].keys()).index(xname)  # dict are ordered as of python 3.7

    yname = parameters[1]
    y = grids_dict["xy"][yname]["values"]
    ylabel = grids_dict["xy"][yname]["name"]
    yidx = list(grids_dict["xy"].keys()).index(yname)

    # create meshgrid
    xx, yy = np.meshgrid(x, y)

    with mpl.rc_context(plot_context):
        for zname in grids_dict["z"]:
            # retrieve grid
            zz = np.array(grids_dict["z"][zname])

            # if greater dimension than 2, aggregate over other dimensions
            if len(zz.shape) > 2:
                dims = [i for i in range(len(zz.shape)) if (i != xidx and i != yidx)]
                zz = np.max(zz, dims)

            f = plt.figure()
            plt.contourf(xx*100., yy*100., uniform_filter(zz, size=3, mode='mirror'), 100)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            cbar = plt.colorbar()
            cbar.ax.set_ylabel(zname)
            plt.tight_layout()
            f.savefig(f"{path}/{zname.replace(' ', '-')}.pdf")
            plt.close(f)


def parse_args():
    parser = argparse.ArgumentParser(description="script making plots.")
    parser.add_argument('-m', '--mode', type=str, required=True, choices=('run', 'avg_run', 'grid'), default='avg_run')
    parser.add_argument('-l', '--logdir', type=str, required=True,
                        help="path to the directory containing the necessary files for plotting, "
                             "('agent': pickled agent and the tensorboard event file., 'grid': json file with results")
    parser.add_argument('-t', '--trajectory', type=bool, default=False,
                        help="whether a trajectory obtained with the trained agent should be plotted (default:True, , only valid if mode is 'agent').")
    parser.add_argument('-gp', '--grid_parameters', type=str, nargs='+', default=['omega', 'zeta'],
                        help="parameters than should be displayed in the grid plot, other dimensions are aggregated using a max operation.")
    return parser.parse_args()


def main_run(plots_path, args, config):
    # load trained agent


    # plot trajectory if needed
    if args.trajectory:
        agent = utils.load_agent(args.logdir)
        trajectory, _, _, _, _, _ = agent.sample_trajectory(1)
        states_repr = agent.environment.system_nn.states_repr()
        plot_trajectory(trajectory, states_repr, path=plots_path)

    # plot tensorboard logs
    event_acc = EventAccumulator(args.logdir)
    event_acc.Reload()
    plot_tb_logs(TB_PLOTS[config["system_name"]], [event_acc], plots_path=plots_path)


def main_avg_run(plots_path, args, config):
    dirs = utils.list_run_dirs(args.logdir)
    # get all the tb event accumulators
    event_accs = []
    for dir in dirs:
        event_acc = EventAccumulator(dir)
        event_acc.Reload()
        event_accs.append(event_acc)

    # plot tensorboard logs
    plot_tb_logs(TB_PLOTS[config["system_name"]], event_accs=event_accs, plots_path=plots_path)


def main_grid(plots_path, args, config):
    results = utils.load_json(os.path.join(args.logdir, "results.json"))
    plot_grids(results, args.grid_parameters, plots_path)


if __name__ == '__main__':
    utils.set_seeds(42)  # necessary for trajectory sampling
    args = parse_args()

    # In every case, create destinatio, directry for plots
    config = utils.load_json(os.path.join(args.logdir, "config.json"))
    plots_path = os.path.join(args.logdir, "plots")
    os.makedirs(plots_path, exist_ok=True)

    if args.mode == "run":
        main_run(plots_path, args, config)
    elif args.mode == "avg_run":
        main_avg_run(plots_path, args, config)
    elif args.mode == "grid":
        main_grid(plots_path, args, config)
    else:
        raise ValueError("Allowed modes are: ")
