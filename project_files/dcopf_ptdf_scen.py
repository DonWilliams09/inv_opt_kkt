import numpy as np
import cvxpy as cp
import re
import json
import os
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
Pd[8] += 60  # Bus 9
Pd[9] += 40  # Bus 10
Pd[12] += 50  # Bus 13
Pd[1] -= 20  # Bus 2
Pd[2] -= 10  # Bus 3

#print("total load demand: ", np.sum(Pd))
Pg_min = gen_data[:, 9]  # Minimum generation
Pg_max = gen_data[:, 8]  # Maximum generation
# Pg_max[0] = 300
# Pg_max[1] = 200
print("total generation capacity: ", np.sum(Pg_max))
cost_coeff = gencost_data[:, 5]  # Linear cost coefficient
print("branch data max capacity: ", branch_data[:, 5])

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


# --- Load Perturbation Loop ---

# Create output directory
os.makedirs("results", exist_ok=True)
output_path = "results/test.json"

# Load existing file or start new list
if os.path.exists(output_path):
    with open(output_path, "r") as f:
        data = json.load(f)
else:
    data = []

scenario = 0
max_iterations = 1000  # Optional stop condition

while True:
    scenario += 1
    if scenario > max_iterations:  # comment this out to run forever
        break

    print(f"\n--- Scenario {scenario} ---")

    # Random demand perturbation (±30% max)
    # noise = 0.3 * (2 * np.random.rand(len(Pd)) - 1)  # uniform in [-0.3, 0.3]
    # Pd_new = Pd * (1 + noise)
    
    # low = Pd
    # high = np.array([1, 90, 175, 110, 50, 70, 1, 1, 100, 50, 35, 50, 70, 70])
    # Pd_new = np.random.randint(low, high)

    #Pd_new = np.random.randint(0, 100, size=Pd.shape) + np.random.randint(0, 100, size=Pd.shape)/100
    Pd_new = Pd + np.random.randint(-10, 10, size=Pd.shape)

    # Pd_new[0] = 0  # slack bus
    # Pd_new[6] = 0  
    # Pd_new[7] = 0    
    
    # # Add random value in [0, 50] to each bus load
    # variation = np.random.uniform(0, 20, size=Pd.shape)
    # Pd_new = Pd + variation
    
    print(f"Total load demand: {np.sum(Pd_new):.2f}")
    
    # Congest some lines artificially
    branch_data_congested = branch_data.copy()
    # branch_data_congested[3, 5] = 50  # Branch 4–5
    # branch_data_congested[4, 5] = 40  # Branch 5–6
    # branch_data_congested[6, 5] = 30  # Branch 4–9
    # branch_data_congested[10, 5] = 35  # Branch 6–11


    # Compute PTDF with new branch limits
    PTDF = compute_PTDF(branch_data_congested, bus_data)

    # Define injection and constraints
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
        print(f"⚠️ Scenario {scenario} infeasible or not optimal: {prob.status}")
        continue

    # If feasible, collect results
    P_inj_opt = gen_to_bus @ Pg.value - Pd_new
    P_inj_reduced_opt = P_inj_opt[1:]
    power_flows = PTDF @ P_inj_reduced_opt
    lower_flow_duals = constraints[3].dual_value
    upper_flow_duals = constraints[4].dual_value
    lambda_slack = constraints[0].dual_value

    lmp = np.zeros(num_buses)
    lmp[0] = lambda_slack
    for i in range(1, num_buses):
        congestion = np.sum(PTDF[:, i-1] * (upper_flow_duals - lower_flow_duals))
        lmp[i] = lambda_slack + congestion

    congested_lines = []
    for i, pf in enumerate(power_flows):
        limit = branch_data_congested[i, 5]
        if abs(pf) >= limit:
            congested_lines.append({
                "line": int(i + 1),
                "flow": float(pf),
                "limit": float(limit)
            })
            
    # Check if any line is congested
    has_congestion = len(congested_lines) > 0
    
    # Save only if congested
    if not has_congestion:
        print(f"⚠️ Scenario {scenario} is feasible but has no congestion — skipping.")
        continue
    
    result = {
        "scenario_id": scenario,
        #"has_congestion": has_congestion,
        "total_load": float(np.sum(Pd_new)),
        "load_vector": Pd_new.tolist(),
        "objective_cost": float(prob.value),
        "generator_dispatch": Pg.value.tolist(),
        "slack_bus_lmp": float(lambda_slack),
        "lmp": lmp.tolist(),
        "congested_lines": congested_lines,
        "power_flows": power_flows.tolist()
    }

    # Append to file
    data.append(result)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"✅ Scenario {scenario} saved.")
