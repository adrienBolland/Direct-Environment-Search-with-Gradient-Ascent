# Direct Environment Search with Gradient Ascent #

Related paper: https://arxiv.org/abs/2006.01738


### Launching the experiments

The `main.py` script can launched with the following arguments:
```
  --config_file Path to the json configuration file
  --experiment What experiment to perform : 'run', 'avg_run', 'grid'
  --workers Number of parallel workers for the experiments 'avg_run' and 'grid'
  --average Number of runs for averaging for the experiments 'avg_run' and 'grid'
```

The `make_plot.py` script can launched with the following arguments:
```
  --mode What experiment to load : 'run', 'avg_run', 'grid'
  --logdir Path to the save files
  --trajectory Whether to simulate a trajectory or not from th environment
  --grid_parameters Pair of parameters over which to plot the gridsearch
```

### Implementing a new environment and policy 

In order to create a new environment, one shall implement a system inheriting from the class 'systems.System'.
The name of the system must be added to the list in 'systems.__init__.py'. 
Equivalently a policy is a class inheriting from 'policies.Policy' whose name is in the list in 'policies.__init__.py'
A configuration file must be created specifying the different arguments of the system and the policy.
