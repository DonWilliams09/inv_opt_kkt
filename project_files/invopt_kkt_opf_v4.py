import numpy as np
import cvxpy as cp
from utils import parse_m_file, compute_PTDF

# Parameters
num_scenarios = 20
perturbation_scale = 0.1
baseMVA = 100

# Load data
bus_data, gen_data, branch_data, gencost_data = parse_m_file('pglib_opf_case14_ieee.m')
num_buses = bus_data.shape[0]
num_generators = gen_data.shape[0]
num_lines = branch_data.shape[0]

# Data Extraction
Pd_base = bus_data[:, 2]
Pg_min = gen_data[:, 9]
Pg_max = gen_data[:, 8]
Pg_max[0] = 300
Pg_max[1] = 200
cost_coeff_true = gencost_data[:, 5]
PTDF = compute_PTDF(branch_data, bus_data)

# Generator incidence matrix
gen_to_bus = np.zeros((num_buses, num_generators))
for i, gen_bus in enumerate(gen_data[:, 0].astype(int) - 1):
    gen_to_bus[gen_bus, i] = 1

# Storage for scenario data
Pg_scenarios = []
lambda_slack_scenarios = []
nu_max_scenarios = []
nu_min_scenarios = []
Pd_scenarios = []

scenario_count = 0
np.random.seed(42)

while scenario_count < num_scenarios:
    Pd_perturbed = Pd_base * (1 + np.random.uniform(-perturbation_scale, perturbation_scale, size=num_buses))
    #Pd_perturbed = np.random.randint(0, 100, size=Pd_base.shape) + np.random.randint(0, 100, size=Pd_base.shape)/100
    Pd_perturbed[0] = 0  # slack bus
    Pd_scenarios.append(Pd_perturbed)

    Pg = cp.Variable(num_generators)
    P_inj = gen_to_bus @ Pg - Pd_perturbed
    P_inj_reduced = P_inj[1:]

    constraints = [
        cp.sum(Pg) == np.sum(Pd_perturbed),
        Pg >= Pg_min,
        Pg <= Pg_max,
        -branch_data[:, 5] <= PTDF @ P_inj_reduced,
        PTDF @ P_inj_reduced <= branch_data[:, 5]
    ]

    objective = cp.Minimize(cp.sum(cp.multiply(cost_coeff_true, Pg)))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.MOSEK)

    if prob.status == 'optimal':
        scenario_count += 1
        Pg_scenarios.append(Pg.value)
        lambda_slack_scenarios.append(constraints[0].dual_value)
        nu_min_scenarios.append(constraints[3].dual_value)
        nu_max_scenarios.append(constraints[4].dual_value)
        print(f"Scenario {scenario_count}: Optimal")
    else:
        print(f"Scenario skipped (infeasible or not optimal)")

Pg_scenarios = np.array(Pg_scenarios)
lambda_slack_scenarios = np.array(lambda_slack_scenarios)
nu_max_scenarios = np.array(nu_max_scenarios)
nu_min_scenarios = np.array(nu_min_scenarios)

# Inverse Optimization (using multiple scenarios)
c = cp.Variable(num_generators, nonneg=True)
loss = 0

for t in range(num_scenarios):
    Pg_inv = cp.Variable(num_generators)
    lambda_slack_inv = cp.Variable()
    nu_min_inv = cp.Variable(num_lines)
    nu_max_inv = cp.Variable(num_lines)

    P_inj_inv = gen_to_bus @ Pg_inv - Pd_scenarios[t]
    P_inj_reduced_inv = P_inj_inv[1:]

    stationarity = []
    for i in range(num_generators):
        congestion_term_inv = (nu_max_inv - nu_min_inv) @ PTDF @ gen_to_bus[1:, i]
        stationarity.append(
            c[i] + lambda_slack_inv + congestion_term_inv == 0
        )

    primal_feasibility = [
        cp.sum(Pg_inv) == np.sum(Pd_scenarios[t]),
        Pg_inv >= Pg_min,
        Pg_inv <= Pg_max,
        -branch_data[:, 5] <= PTDF @ P_inj_reduced_inv,
        PTDF @ P_inj_reduced_inv <= branch_data[:, 5]
    ]
    
    dual_feasibility = [
        # mu_min_inv >= 0,
        # mu_max >= 0,
        nu_min_inv >= 0,
        nu_max_inv >= 0
    ]

    loss += cp.norm(Pg_inv - Pg_scenarios[t], 2)**2
    loss += cp.norm(lambda_slack_inv - lambda_slack_scenarios[t], 2)**2
    loss += cp.norm(nu_max_inv - nu_max_scenarios[t], 2)**2
    loss += cp.norm(nu_min_inv - nu_min_scenarios[t], 2)**2

    constraints += stationarity + primal_feasibility

#constraints.append(c[2:5] == 0)
loss *= 1 / num_scenarios

inv_prob = cp.Problem(cp.Minimize(loss), constraints)
inv_prob.solve(solver=cp.MOSEK, verbose=True)

if inv_prob.status == 'optimal':
    print("Inferred cost coefficients: ", c.value)
else:
    print("Inverse problem not optimal: ", inv_prob.status)
