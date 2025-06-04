import numpy as np
import cvxpy as cp
from utils import *

np.set_printoptions(precision=3, suppress=True, floatmode='fixed')

# Base MVA
baseMVA = 100

# Load case data manually
bus_data, gen_data, branch_data, gencost_data = parse_m_file('pglib_opf_case14_ieee.m')

num_buses = bus_data.shape[0]
num_generators = gen_data.shape[0]
num_lines = branch_data.shape[0]

# Data Extraction
Pd = bus_data[:, 2]  # Load demand
Pg_min = gen_data[:, 9]  # Minimum generation
Pg_max = gen_data[:, 8]  # Maximum generation
#Pg_max[2:5] = 50 # Constant generation capacity for generators 3-5 : 10 MW
Pg_max[0] = 300
Pg_max[1] = 200
print("Total generation capacity: ", np.sum(Pg_max))
cost_coeff = gencost_data[:, 5]  # Linear cost coefficient
print("Cost coefficients: ", cost_coeff)


# Compute PTDF
PTDF = compute_PTDF(branch_data, bus_data)

# Define optimization variables
Pg = cp.Variable(num_generators)

# Create generator-to-bus incidence matrix
gen_to_bus = np.zeros((num_buses, num_generators))
for i, gen_bus in enumerate(gen_data[:, 0].astype(int) - 1):
    gen_to_bus[gen_bus, i] = 1

# Objective: Minimize generation cost
objective = cp.Minimize(cp.sum(cp.multiply(cost_coeff, Pg)))

# --- Congestion Scenario Experiment ---
print("\n\n========== FORCED CONGESTION SCENARIO ==========")

# Deep copy to avoid overwriting original
Pd_new = Pd.copy()

# Step 1: Drastically increase load at specific buses
Pd_new *= 1.2
# Pd_new[1] += 80
# # Pd_new[3] += 5
# Pd_new[12] += 5
# Pd_new[13] += 7
# # Pd_new[10] += 5

print(f"New total load demand: {np.sum(Pd_new):.2f} MW")
print("Perturbed Load = ", Pd_new)

# Step 2: Reduce line capacity to force congestion
branch_data_congested = branch_data.copy()
# Example: reduce capacity on lines 3 and 6 (indices 2 and 5)
branch_data_congested[:, 5] *= 0.6  # reduce limits by 45%
branch_data_congested[0, 5] = 150  # set a specific limit
branch_data_congested[15, 5] = 100  # set a specific limit
print("Modified line limits: ", branch_data_congested[:, 5])


# Recompute PTDF with same topology (line reactances unchanged)
PTDF = compute_PTDF(branch_data_congested, bus_data)

# Define new power injection and constraints
P_inj = gen_to_bus @ Pg - Pd_new
P_inj_reduced = P_inj[1:]
power_balance = cp.sum(Pg) == np.sum(Pd_new)

constraints = [
    power_balance,
    Pg >= Pg_min,
    Pg <= Pg_max,
    -branch_data_congested[:, 5] <= PTDF @ P_inj_reduced,
    PTDF @ P_inj_reduced <= branch_data_congested[:, 5]
]

prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.MOSEK)

if prob.status != 'optimal':
    print(f"⚠️ No optimal == Problem Status: {prob.status}")
else:
    # Post-processing
    P_inj_opt = gen_to_bus @ Pg.value - Pd_new
    P_inj_reduced_opt = P_inj_opt[1:]
    power_flows = PTDF @ P_inj_reduced_opt
    print("Power flows: ", power_flows)

    lower_flow_duals = constraints[3].dual_value
    print("Lower flow duals: ", lower_flow_duals)
    upper_flow_duals = constraints[4].dual_value
    print("Upper flow duals: ", upper_flow_duals)
    flow_duals = upper_flow_duals - lower_flow_duals
    print("Flow duals: ", flow_duals)
    lambda_slack = constraints[0].dual_value
    print("Slack bus dual: ", lambda_slack)

    print(f"\nOptimal Cost: {prob.value:.2f} $ \n")  
    print("Optimal Generation Dispatch (MW):")
    for i, val in enumerate(Pg.value):
        print(f"Generator {i+1}: {val:.2f} MW")

    
    print("Max Line Flow = {:.2f} MW".format(np.max(np.abs(power_flows))))

    print("\nActive Congestion (lines near limits):")
    for i, pf in enumerate(power_flows):
        limit = branch_data_congested[i, 5]
        #if abs(pf) >= 0.98 * limit:
        print(f"  Line {i+1}: {pf:.2f} MW --> {(pf/limit)*100:.2f}% (Limit: {limit:.2f} MW)")

    print("\nDuals on line flow constraints (non-zero = active):")
    for i in range(num_lines):
        if abs(lower_flow_duals[i]) > 1e-4 or abs(upper_flow_duals[i]) > 1e-4:
            print(f"  Line {i+1}: λ- = {lower_flow_duals[i]:.3f}, λ+ = {upper_flow_duals[i]:.3f}")

    # # Calculate LMPs
    # print("\nSlack LMP = {:.2f} $/MWh".format(lambda_slack))
    # lmp = np.zeros(num_buses)
    # lmp[0] = lambda_slack
    # for i in range(1, num_buses):
    #     congestion = np.sum(PTDF[:, i-1] * (upper_flow_duals - lower_flow_duals))
    #     lmp[i] = lambda_slack + congestion
    # print("\nLMPs:")
    # for i in range(num_buses):
    #     print(f"  Bus {i+1}: LMP = {lmp[i]:.2f}")


# --- Inverse Optimisation ---
Pg_observed = Pg.value
#flow_duals_observed = flow_duals
lower_flow_duals_observed = lower_flow_duals
upper_flow_duals_observed = upper_flow_duals
lambda_slack_observed = lambda_slack
print("Observed generation: ", Pg_observed)
print("Observed total generation: ", np.sum(Pg_observed))
# Cost vector to learn
c_min = 0
c_max = 100
c = cp.Variable(num_generators, nonneg=True)   

# Variables of the upper (forward) problem
Pg_inv = cp.Variable(num_generators)
lmp_inv = cp.Variable(num_buses)
mu_min = cp.Variable(num_generators)
mu_max = cp.Variable(num_generators)
flow_duals_inv = cp.Variable(num_lines)
nu_min = cp.Variable(num_lines)
nu_max = cp.Variable(num_lines)
lambda_slack_inv = cp.Variable()  # Lagrange multiplier for power balance

P_inj_inv = gen_to_bus @ Pg_inv - Pd_new
P_inj_reduced_inv = P_inj_inv[1:]

# Define the KKT conditions

# Cost coefficients
c = cp.Variable(num_generators, nonneg=True)

# Stationarity conditions
stationarity = []
for i in range(num_generators):
    congestion_term_inv = 0
    for line in range(num_lines):
        # Partial derivative of P_inj_reduced w.r.t Pg[i] is gen_to_bus[1:, i]
        congestion_term_inv += (nu_max[line] - nu_min[line]) * PTDF[line] @ gen_to_bus[1:, i]
    stationarity.append(
        c[i] + lambda_slack_inv + mu_max[i] - mu_min[i] + congestion_term_inv == 0
    )

# Primal feasibility conditions 

primal_feasibility = [
    cp.sum(Pg_inv) == np.sum(Pd_new),
    Pg_inv >= Pg_min,
    Pg_inv <= Pg_max,
    -branch_data_congested[:, 5] <= PTDF @ P_inj_reduced_inv,
    PTDF @ P_inj_reduced_inv <= branch_data_congested[:, 5]
]

# Dual feasibility conditions -- Only for inegality constraints
dual_feasibility = [
    mu_min >= 0,
    mu_max >= 0,
    nu_min >= 0,
    nu_max >= 0
]

# Complementarity Slackness conditions
# Big-M method for linearization --> Don't forget to refine it when the code run well
Big_M_Pg = 500
Big_M_Flow = 750

# Binary variables for complementary slackness
z_mu_min = cp.Variable(mu_min.shape, boolean=True)
z_mu_max = cp.Variable(mu_max.shape, boolean=True)
z_nu_min = cp.Variable(nu_min.shape, boolean=True)
z_nu_max = cp.Variable(nu_max.shape, boolean=True)

complementary = []

# Complementary slackness conditions linearization (Big-M)
# mu_min * (Pg_min - Pg) == 0
complementary += [
    Pg_min - Pg_inv <= Big_M_Pg * z_mu_min,
    mu_min <= Big_M_Pg * (1 - z_mu_min)
]

# mu_max * (Pg - Pg_max) == 0
complementary += [
    Pg_inv - Pg_max <= Big_M_Pg * z_mu_max,
    mu_max <= Big_M_Pg * (1 - z_mu_max)
]

# nu_min * (- branch_data_congested[:, 5] - PTDF @ P_inj_reduced) == 0
complementary += [
    - branch_data_congested[:, 5] - PTDF @ P_inj_reduced_inv <= Big_M_Flow * z_nu_min,
    nu_min <= Big_M_Flow * (1 - z_nu_min)
]

# nu_max * (PTDF @ P_inj_reduced - branch_data_congested[:, 5]) == 0
complementary += [
    PTDF @ P_inj_reduced_inv - branch_data_congested[:, 5] <= Big_M_Flow * z_nu_max,
    nu_max <= Big_M_Flow * (1 - z_nu_max)
]

# Define Incerse Problem
loss =    cp.norm(Pg_inv - Pg_observed, 2)**2 + cp.norm(lambda_slack_inv - lambda_slack_observed, 2)**2 + cp.norm(nu_max - upper_flow_duals_observed, 2)**2 + cp.norm(nu_min - lower_flow_duals_observed, 2)**2


constraints = [
    *stationarity,
    *primal_feasibility,
    *dual_feasibility,
    *complementary,
    c[2:5] == 0,
]

inv_prob = cp.Problem(cp.Minimize(loss), constraints)
inv_prob.solve(solver=cp.MOSEK, verbose=True)
#print_problem_summary(inv_prob)


if inv_prob.status != 'optimal':
    print(f"⚠️ No optimal == Problem Status: {inv_prob.status}")
else:   
    print("Inferred optimal cost: ", inv_prob.value)
    print("Inferred optimal generation: ", Pg.value)
    print("Inferred total generation: ", np.sum(Pg.value))
    print("Inferred cost coefficients: ", c.value)
    print("Inferred lambda: ", lambda_slack_inv.value)
    print("Inferred flow duals: ", nu_max.value - nu_min.value) 
    print("Mu min: ", mu_min.value)
    print("Mu max: ", mu_max.value)
    print("Nu min: ", nu_min.value)
    print("Nu max: ", nu_max.value)
    
    lmp = np.zeros(num_buses)
    lmp[0] = lambda_slack_inv.value
    for i in range(1, num_buses):
        congestion = np.sum(PTDF[:, i-1] @ (nu_max.value - nu_min.value))
        lmp[i] = lambda_slack + congestion
    print("\n Inferred LMPs:")
    for i in range(num_buses):
        print(f"  Bus {i+1}: Inferred LMP = {lmp[i]:.2f}")
    

