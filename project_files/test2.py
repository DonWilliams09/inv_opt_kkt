import numpy as np
import cvxpy as cp
from oct2py import Oct2Py

# Start Octave session and load MATPOWER case
oc = Oct2Py()
oc.eval("addpath('/Users/don_williams09/Downloads/Bi_Level_Opt')")  # Adjust to your path
mpc = oc.case300()

# Extract MATPOWER data
bus_data = mpc['bus']
gen_data = mpc['gen']
branch_data = mpc['branch']
gencost_data = mpc['gencost']

# Detect slack bus (type == 3)
slack_bus = int(bus_data[bus_data[:, 1] == 3, 0][0])

# Map bus IDs to indices
bus_ids = bus_data[:, 0].astype(int)
id_to_index = {bus_id: idx for idx, bus_id in enumerate(bus_ids)}
num_buses = len(bus_ids)
num_generators = gen_data.shape[0]
num_lines = branch_data.shape[0]

# Data
Pd = bus_data[:, 2]              # Demand
Pg_min = gen_data[:, 9]         # Min gen
Pg_max = gen_data[:, 8]         # Max gen
cost_coeff = gencost_data[:, 5] # Cost linear term

# Compute PTDF
def compute_PTDF(branch_data, bus_data, slack_bus):
    num_lines = branch_data.shape[0]
    bus_ids = bus_data[:, 0].astype(int)
    id_to_index = {bus_id: idx for idx, bus_id in enumerate(bus_ids)}
    num_buses = len(bus_ids)

    B = np.zeros((num_buses, num_buses))
    for f_id, t_id, x in zip(branch_data[:, 0].astype(int),
                             branch_data[:, 1].astype(int),
                             branch_data[:, 3]):
        f = id_to_index[f_id]
        t = id_to_index[t_id]
        B[f, t] = -1 / x
        B[t, f] = -1 / x
    for i in range(num_buses):
        B[i, i] = -np.sum(B[i, :])
    
    slack_index = id_to_index[slack_bus]
    B_reduced = np.delete(np.delete(B, slack_index, axis=0), slack_index, axis=1)
    B_inv = np.linalg.pinv(B_reduced)

    B_line = np.zeros((num_lines, num_buses))
    for idx, (f_id, t_id, x) in enumerate(zip(branch_data[:, 0].astype(int),
                                              branch_data[:, 1].astype(int),
                                              branch_data[:, 3])):
        f = id_to_index[f_id]
        t = id_to_index[t_id]
        B_line[idx, f] = 1 / x
        B_line[idx, t] = -1 / x

    B_line_reduced = np.delete(B_line, slack_index, axis=1)
    PTDF = B_line_reduced @ B_inv
    return PTDF, id_to_index

PTDF, id_to_index = compute_PTDF(branch_data, bus_data, slack_bus)

# Generator-to-bus incidence matrix
gen_to_bus = np.zeros((num_buses, num_generators))
for i, gen_bus_id in enumerate(gen_data[:, 0].astype(int)):
    gen_to_bus[id_to_index[gen_bus_id], i] = 1

# Optimization variables
Pg = cp.Variable(num_generators)

# Net injection (exclude slack)
P_inj = gen_to_bus @ Pg - Pd
slack_index = id_to_index[slack_bus]
P_inj_reduced = np.delete(P_inj, slack_index)

# Objective and constraints
objective = cp.Minimize(cp.sum(cp.multiply(cost_coeff, Pg)))
constraints = [
    cp.sum(Pg) == np.sum(Pd),
    Pg >= Pg_min,
    Pg <= Pg_max,
    -branch_data[:, 5] <= PTDF @ P_inj_reduced,
    PTDF @ P_inj_reduced <= branch_data[:, 5]
]

# Solve the problem
prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.MOSEK, verbose=True)

# Output
if prob.status == cp.OPTIMAL:
    print(f"✅ DC-OPF solved: Optimal cost = {prob.value:.2f}")
    print("Sample generation dispatch (first 10):", Pg.value[:10])
else:
    print("❌ DC-OPF problem is not optimal. Status:", prob.status)
