import numpy as np
from pyomo.environ import *
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt

import utils

MG_CONFIG = {
    "config_file": "config/config_1.json",
    "experiment": "run",
}


def plot_opt(model):
    plt.figure()
    plt.plot([model.load_shed[i].value for i in model.Periods])
    plt.plot([model.curtail[i].value for i in model.Periods])

    f, ax = plt.subplots(4, 1, sharex=True)
    ax[0].plot([model.soc[i].value for i in model.Periods_plus_one], label="State of charge")
    ax[1].plot([model.charge_power[i].value for i in model.Periods], label="Charge power")
    ax[1].plot([model.discharge_power[i].value for i in model.Periods], label="Discharge")
    ax[2].plot([model.gen[i].value for i in model.Periods], label="Generator")
    ax[3].plot([model.total_load[i] for i in model.Periods], label="Load")
    ax[3].plot([model.epv_cf[i] * model.max_epv_cap.value for i in model.Periods], label="Pv production")
    [a.legend() for a in ax]

    plt.figure()
    plt.plot(np.cumsum(np.array([model.shed_cost[i].value for i in model.Periods])), label="Shed")
    plt.plot(np.cumsum(np.array([model.curtail_cost[i].value for i in model.Periods])), label="curts")
    plt.plot(np.cumsum(np.array([model.fuel_cost[i].value for i in model.Periods])), label="gen_cost")
    plt.legend()


def opimization(dem_prof, epv_cf_prof, charge_eff, discharge_eff, power_rating, fuel_price, load_curtail_price,
                load_shed_price, pv_cost, bat_cost, gen_cost, inv_rate, years, bat_size=None,
                pv_size=None, gen_size=None):
    model = ConcreteModel()

    # SETS
    horizon = len(dem_prof)
    model.Periods = Set(initialize=range(horizon))
    model.Periods_plus_one = Set(initialize=range(horizon + 1))
    # PARAMS

    model.total_load = Param(model.Periods, initialize=dem_prof)
    model.epv_cf = Param(model.Periods, initialize=epv_cf_prof)

    # VARIABLES
    model.epv_prod = Var(model.Periods, within=NonNegativeReals)  # Auxiliary variable
    model.load_shed = Var(model.Periods, within=NonNegativeReals)  # Auxiliary variable
    model.curtail = Var(model.Periods, within=NonNegativeReals)  # Auxiliary variable

    model.charge_power = Var(model.Periods, within=NonNegativeReals)  # Decision variable
    model.discharge_power = Var(model.Periods, within=NonNegativeReals)  # Decision variable
    model.gen = Var(model.Periods, within=NonNegativeReals)  # Decision variable
    model.soc = Var(model.Periods_plus_one, within=NonNegativeReals)  # State variable

    model.fuel_cost = Var(model.Periods, within=NonNegativeReals)
    model.curtail_cost = Var(model.Periods, within=NonNegativeReals)
    model.shed_cost = Var(model.Periods, within=NonNegativeReals)
    model.inv_cost = Var(within=NonNegativeReals)
    model.bin = Var(model.Periods, within=Binary)

    model.max_storage_cap = Var(within=NonNegativeReals)
    model.max_epv_cap = Var(within=NonNegativeReals)  # Auxiliary variabl
    model.max_gen_cap = Var(within=NonNegativeReals)
    model.total_cost = Var(within=NonNegativeReals)
    if (pv_size is not None) and (bat_size is not None) and (gen_size is not None):
        model.max_epv_cap.fix(pv_size)
        model.max_storage_cap.fix(bat_size)
        model.max_gen_cap.fix(gen_size)

    # CONSTRAINTS
    def energy_balance(m, p):
        return m.discharge_power[p] + m.epv_prod[p] + m.gen[p] + m.load_shed[p] == m.charge_power[p] + m.total_load[
            p] + m.curtail[p]

    def storage_evolution(m, p):
        return m.soc[p + 1] == m.soc[p] + (m.charge_power[p] * charge_eff - m.discharge_power[p] / discharge_eff)

    def max_soc(m, p):
        return m.soc[p] <= m.max_storage_cap

    def charge_constraint(m, p):
        return m.charge_power[p] <= m.bin[p] * 1000

    def discharge_constraint(m, p):
        return m.discharge_power[p] <= (1 - m.bin[p]) * 1000

    def max_charge_power(m, p):
        return m.charge_power[p] <= m.max_storage_cap * power_rating

    def max_discharge_power(m, p):
        return m.discharge_power[p] <= m.max_storage_cap * power_rating

    def init_soc(m, p):
        if p == 0:
            return m.soc[p] == m.max_storage_cap / 2.
        else:
            return Constraint.Skip

    def max_epv(m, p):
        return m.epv_prod[p] <= m.max_epv_cap

    def epv_production(m, p):
        return m.epv_prod[p] == m.max_epv_cap * m.epv_cf[p]

    def generation_conv(m, p):
        return m.gen[p] <= m.max_gen_cap

    def fuel_gen_cost(m, p):
        return m.fuel_cost[p] == fuel_price * m.gen[p]

    def curtail_gen_cost(m, p):
        return m.curtail_cost[p] == load_curtail_price * m.curtail[p]

    def shed_load_cost(m, p):
        return m.shed_cost[p] == load_shed_price * m.load_shed[p]

    def investment_cost(m):
        return m.inv_cost == (pv_cost * m.max_epv_cap + bat_cost * m.max_storage_cap + gen_cost * m.max_gen_cap) * (
                    inv_rate * (1 + inv_rate) ** years) / ((1 + inv_rate) ** years - 1) / 8760

    def total_cost_fun(m):
        return m.total_cost == sum((m.inv_cost +
                                    m.fuel_cost[p] + m.curtail_cost[p] + m.shed_cost[p]) * 8760 / horizon for p in
                                   m.Periods)

    def objective(m):
        return m.total_cost

    model.energy_balance = Constraint(model.Periods, rule=energy_balance)
    model.storage_evolution = Constraint(model.Periods, rule=storage_evolution)
    model.max_soc = Constraint(model.Periods, rule=max_soc)
    model.max_discharge_power = Constraint(model.Periods, rule=max_discharge_power)
    model.max_charge_power = Constraint(model.Periods, rule=max_charge_power)
    model.discharge_constraint = Constraint(model.Periods, rule=discharge_constraint)
    model.charge_constraint = Constraint(model.Periods, rule=charge_constraint)
    model.init_soc = Constraint(model.Periods, rule=init_soc)
    model.max_epv = Constraint(model.Periods, rule=max_epv)
    model.epv_production = Constraint(model.Periods, rule=epv_production)
    model.generation_conv = Constraint(model.Periods, rule=generation_conv)
    model.fuel_gen_cost = Constraint(model.Periods, rule=fuel_gen_cost)
    model.curtail_gen_cost = Constraint(model.Periods, rule=curtail_gen_cost)
    model.shed_load_cost = Constraint(model.Periods, rule=shed_load_cost)
    model.investment_cost = Constraint(rule=investment_cost)
    model.total_cost_fun = Constraint(rule=total_cost_fun)
    model.obj = Objective(rule=objective, sense=minimize)

    solver = SolverFactory("cplex")
    results = solver.solve(model)

    return model


if __name__ == '__main__':
    plot_flag = False
    args = utils.load_json(MG_CONFIG["config_file"])

    system_args = args["system_args"]
    dem_size = system_args["dem_size"]
    pv_size = system_args["pv_size"]
    bat_size = system_args["bat_size"]
    gen_size = system_args["gen_size"]
    power_rating = system_args["power_rating"]
    bat_cost = system_args["bat_cost"]
    pv_cost = system_args["pv_cost"]
    gen_cost = system_args["gen_cost"]
    inv_rate = system_args["inv_rate"]
    inv_years = system_args["inv_years"]
    fuel_price = system_args["fuel_price"]
    load_shed_price = system_args["load_shed_price"]
    load_curtail_price = system_args["load_curtail_price"]
    charge_eff, discharge_eff = system_args["charge_eff"], system_args["discharge_eff"]
    horizon = system_args["horizon"]

    pv_avg_prod = np.array([0.00000001e+00, 0.00000001e+00, 0.00000001e+00, 0.00000001e+00,
                            0.00000001e+00, 0.00000001e+00, 0.00000001e+00, 0.00000001e+00,
                            0.00000001e+00, 4.62232374e-02, 8.89720101e-02, 1.22127062e-01,
                            1.41992336e-01, 1.49666484e-01, 1.43378674e-01, 1.20629623e-01,
                            8.71089652e-02, 4.64848134e-02, 1.84307861e-17, 0.00000001e+00,
                            0.00000001e+00, 0.00000001e+00, 0.00000001e+00, 0.00000001e+00])

    dem_avg = np.array([0.3457438, 0.32335429, 0.309672, 0.29759948, 0.28587788,
                        0.27293944, 0.24240862, 0.22680175, 0.23042503, 0.23326265,
                        0.23884741, 0.24825482, 0.25547133, 0.26739509, 0.27287241,
                        0.27219202, 0.2706911, 0.29403735, 0.42060912, 0.53479381,
                        0.5502525, 0.5267475, 0.46403763, 0.39285948])

    dist_std = np.array([0.05542831, 0.05022998, 0.0432726, 0.03978419, 0.03952021,
                         0.03775034, 0.03728352, 0.03621157, 0.04035931, 0.04320152,
                         0.04408169, 0.04740461, 0.04239965, 0.04087229, 0.04240869,
                         0.04717433, 0.0436305, 0.04424234, 0.08158905, 0.06022856,
                         0.0553013, 0.05767294, 0.06095378, 0.05918214])

    dem_prof = {}
    epv_cf_prof = {}
    for t in range(horizon):
        dem_prof[t] = dem_avg[t % 24] * dem_size + np.random.normal(0., dist_std[t % 24])
        epv_cf_prof[t] = pv_avg_prod[t % 24]

    model = opimization(dem_prof, epv_cf_prof, charge_eff, discharge_eff, power_rating, fuel_price, load_curtail_price,
                        load_shed_price, pv_cost,
                        bat_cost, gen_cost, inv_rate, inv_years, bat_size=None, pv_size=None, gen_size=None)

    print("Size of PV: %.2f, Bat:%.2f, Gen:%.2f" % (
        model.max_epv_cap.value, model.max_storage_cap.value, model.max_gen_cap.value))

    print("Total cost: %.2f" % model.total_cost.value)

    if plot_flag: plot_opt(model)

    aux = []
    for p in model.Periods:
        r = -(model.inv_cost.value + model.fuel_cost[p].value + model.curtail_cost[p].value + model.shed_cost[
            p].value) * 8760 / horizon
        X_std = (r - (-5000)) / (0 - (-5000))
        rew_scaled = X_std * (1 - 0) + 0
        aux.append(rew_scaled)

    sc = sum(aux)
    print("Scaled cost ", sc)
