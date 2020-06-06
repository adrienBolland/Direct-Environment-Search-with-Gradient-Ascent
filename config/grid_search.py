import numpy as np


def msd_grid(nb_samples=15, **kwargs):
    omega_val = np.linspace(kwargs["system_args"]["omega_interval"][0],
                            kwargs["system_args"]["omega_interval"][1], nb_samples + 2)[1:-1]
    zeta_val = np.linspace(kwargs["system_args"]["zeta_interval"][0],
                           kwargs["system_args"]["zeta_interval"][1], nb_samples + 2)[1:-1]

    return {"parameters": {"omega": {"values": omega_val.tolist(),
                                     "name": "$\omega$"},
                           "zeta": {"values": zeta_val.tolist(),
                                    "name": "$\zeta$"},
                           },
            "grid": np.zeros((len(omega_val), len(zeta_val))),
            "idx": [(o, z) for (o, _) in enumerate(omega_val) for (z, _) in enumerate(zeta_val)],
            "list": [{"omega": omega,
                      "zeta": zeta,
                      "constant_param": kwargs["system_args"]["target_parameters_reward"]} for omega in omega_val for zeta in zeta_val]
            }


def mg_grid(nb_samples=15, **kwargs):
    bat_val = np.linspace(kwargs["min_bat_val"], kwargs["max_bat_val"], nb_samples + 2)[1:-1]
    pv_val = np.linspace(kwargs["min_pv_val"], kwargs["max_pv_val"], nb_samples + 2)[1:-1]
    return {"parameters": {"bat": {"values": bat_val.tolist(),
                                     "name": "$\overline{SoC}$"},
                           "pv": {"values": pv_val.tolist(),
                                    "name": "$\overline{P^{PV}}$"},
                           },
            "grid": np.zeros((len(bat_val), len(pv_val))),
            "idx": [(o, z) for (o, _) in enumerate(bat_val) for (z, _) in enumerate(pv_val)],
            "list": [{"bat": bat,
                      "pv": pv} for bat in bat_val for pv in pv_val]
            }
