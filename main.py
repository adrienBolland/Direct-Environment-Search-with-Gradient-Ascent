from experiments.disretization_search import grid_optimization
from experiments.system_search import direct_search, multiple_direct_search

import utils
import os
import argparse

DEFAULT_TASK = 'run'
DEFAULT_WORKERS = 8
DEFAULT_AVG = 20


def parse_args():
    parser = argparse.ArgumentParser(description='script launching direct MDP search experiments.')

    parser.add_argument('-conf', '--config_file', type=str, required=True)
    parser.add_argument('-exp', '--experiment', type=str, choices=('run', 'grid', 'avg_run'), default=DEFAULT_TASK)
    parser.add_argument('-w', '--workers', type=int, default=DEFAULT_WORKERS)
    parser.add_argument('-avg', '--average', type=int, default=DEFAULT_AVG)
    parsed_args = vars(parser.parse_args())

    return parsed_args


def run(log_path, **kwargs):
    # save config
    utils.save_config(log_path, kwargs)

    # train
    direct_search(log_path, **kwargs)


def avg_run(nb_run, nb_workers, log_path, **kwargs):
    # save config
    utils.save_config(log_path, kwargs)

    # perform multiple runs
    multiple_direct_search(log_path, nb_run, nb_workers, **kwargs)


def grid(nb_run, nb_processes, log_path, **kwargs):
    # save config
    utils.save_config(log_path, kwargs)

    # optimize and save
    results = grid_optimization(nb_run, nb_processes, **kwargs)
    utils.save_json(os.path.join(log_path, 'results.json'), results)


if __name__ == '__main__':

    # parse arguments
    config = parse_args()

    # load json default arguments
    args = utils.load_json(config['config_file'])

    # seed for reproducibility
    utils.set_seeds(args['seed'])

    # logging
    model_name, logdir = args.pop('model_name'), args.pop('logdir')
    v = utils.get_version(name=model_name, logdir=logdir)
    log_path = os.path.join(logdir, f'{model_name}-v{v}')

    exp = config.pop('experiment')
    if exp == 'run':
        run(log_path, **args)

    elif exp == 'grid':
        nb_processes = config.pop('workers')
        nb_run = config.pop('average')
        grid(nb_run, nb_processes, log_path, **args)

    elif exp == 'avg_run':
        nb_processes = config.pop('workers')
        nb_run = config.pop('average')
        avg_run(nb_run, nb_processes, log_path, **args)
