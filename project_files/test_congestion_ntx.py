import numpy as np
import cvxpy as cp
import re
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
# Pg_max[0] = 200
# Pg_min[1] = 70
#print("Total generation capacity: ", np.sum(Pg_max))
cost_coeff = gencost_data[:, 5]  # Linear cost coefficient
#cost_coeff[2:5] = 5
#print("Cost coefficients: ", cost_coeff)


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

# check = True
# while check:

#Pd_new = np.random.randint(0, 100, size=Pd.shape) + np.random.randint(0, 100, size=Pd.shape)/100
Pd_new = Pd.copy()
Pd_new[8] += 60  # Bus 9
Pd_new[9] += 40  # Bus 10
Pd_new[12] += 50  # Bus 13
Pd_new[1] -= 20  # Bus 2
Pd_new[2] -= 10  # Bus 3


print(f"New total load demand: {np.sum(Pd_new):.2f} MW")
print("Perturbed Load = ", Pd_new)

# Step 2: Reduce line capacity to force congestion
branch_data_congested = branch_data.copy()
# branch_data_congested[3, 5] = 50  # Branch 4–5
# branch_data_congested[4, 5] = 40  # Branch 5–6
# branch_data_congested[6, 5] = 30  # Branch 4–9
# branch_data_congested[10, 5] = 35  # Branch 6–11

# branch_data_congested[15, 5] = 100  # set a specific limit
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
prob.solve(solver=cp.MOSEK, verbose = True)
# if prob.status == 'optimal':
#     check = False
#     continue

if prob.status != 'optimal':
    print(f"⚠️ No optimal == Problem Status: {prob.status}")
else:
    # Post-processing
    P_inj_opt = gen_to_bus @ Pg.value - Pd_new
    P_inj_reduced_opt = P_inj_opt[1:]
    power_flows = PTDF @ P_inj_reduced_opt

    lower_flow_duals = constraints[3].dual_value
    print("Lower flow duals: ", lower_flow_duals)
    upper_flow_duals = constraints[4].dual_value
    print("Upper flow duals: ", upper_flow_duals)
    lambda_slack = constraints[0].dual_value
    print("Slack bus dual: ", lambda_slack)

    lmp = np.zeros(num_buses)
    lmp[0] = lambda_slack
    for i in range(1, num_buses):
        congestion = np.sum(PTDF[:, i-1] * (upper_flow_duals - lower_flow_duals))
        lmp[i] = lambda_slack + congestion

    print(f"\nOptimal Cost: {prob.value:.2f} $ \n")  
    print("Optimal Generation Dispatch (MW):")
    for i, val in enumerate(Pg.value):
        print(f"Generator {i+1}: {val:.2f} MW")

    print("\nSlack LMP = {:.2f} $/MWh".format(lambda_slack))
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

    print("\nLMPs:")
    for i in range(num_buses):
        print(f"  Bus {i+1}: LMP = {lmp[i]:.2f}")

# To visualize the network (optional):
plot_power_network(power_flows, lmp, branch_data_congested)
