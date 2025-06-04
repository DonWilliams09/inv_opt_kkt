import numpy as np
import cvxpy as cp
import re
from oct2py import Oct2Py

np.set_printoptions(precision=3, suppress=True, floatmode='fixed')

# Start Octave session
oc = Oct2Py()

# Load the MATPOWER case file
oc.eval("addpath('/Users/don_williams09/Downloads/Bi_Level_Opt/test_cases')")  # Adjust path to the folder containing the file
mpc = oc.case1354()

# Extract data

bus_data = mpc['bus']
gen_data = mpc['gen']
branch_data = mpc['branch']
gencost_data = mpc['gencost']

slack = int(bus_data[bus_data[:, 1] == 3, 0][0])
print(f"Slack bus: {slack}")

# Show dimensions or samples
print(f"Bus data shape : {bus_data.shape} ")
print(f"Generator data shape : {gen_data.shape}")
print(f"Branch data shape : {branch_data.shape} ")
print(f"Gencost data shape : {gencost_data.shape}")

num_buses = bus_data.shape[0]
num_generators = gen_data.shape[0]
num_lines = branch_data.shape[0]
num_branches = branch_data.shape[0]

# Data Extraction
Pd_base = bus_data[:, 2]
Pg_min = gen_data[:, 9]
Pg_max = gen_data[:, 8]
cost_coeff_true = gencost_data[:, 5]

#branch_data_congested = branch_data.copy()
#branch_data_congested[:, 5] *= 0.7  # reduce limits by 30%

# PTDF = compute_PTDF(branch_data, bus_data)

def compute_PTDF(branch_data, bus_data, slack_bus=slack):
    """ Compute PTDF matrix given branch and bus data (MATPOWER format) """

    # Map bus IDs to matrix indices
    bus_ids = bus_data[:, 0].astype(int)
    id_to_index = {bus_id: idx for idx, bus_id in enumerate(bus_ids)}

    num_buses = len(bus_ids)
    num_lines = branch_data.shape[0]

    # Admittance matrix (B)
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

    # Remove slack bus row/col
    # slack_index = id_to_index[slack_bus] if slack_bus in id_to_index else 0
    slack_index = 0
    print(f"Slack index: {slack_index}")
    B_reduced = np.delete(np.delete(B, slack_index, axis=0), slack_index, axis=1)
    B_inv = np.linalg.pinv(B_reduced)

    # Compute B_line
    B_line = np.zeros((num_lines, num_buses))
    for idx, (f_id, t_id, x) in enumerate(zip(branch_data[:, 0].astype(int),
                                              branch_data[:, 1].astype(int),
                                              branch_data[:, 3])):
        f = id_to_index[f_id]
        t = id_to_index[t_id]
        B_line[idx, f] = 1 / x
        B_line[idx, t] = -1 / x

    # Remove slack column
    B_line_reduced = np.delete(B_line, slack_index, axis=1)

    # Compute PTDF
    PTDF = B_line_reduced @ B_inv

    return PTDF,id_to_index


# Base MVA
baseMVA = 100


num_buses = bus_data.shape[0]
num_generators = gen_data.shape[0]
num_lines = branch_data.shape[0]


# Data Extraction
Pd = bus_data[:, 2]  # Load demand
Pg_min = gen_data[:, 9]  # Minimum generation
Pg_max = gen_data[:, 8]  # Maximum generation
cost_coeff = gencost_data[:, 5]  # Linear cost coefficient


# Compute PTDF
PTDF,id_to_index = compute_PTDF(branch_data, bus_data)

# Define optimization variables
Pg = cp.Variable(num_generators)

# Generator_to_bus incidence matrix
gen_to_bus = np.zeros((num_buses, num_generators))
for i, gen_bus_id in enumerate(gen_data[:, 0].astype(int)):
    bus_index = id_to_index[gen_bus_id]
    gen_to_bus[bus_index, i] = 1

# Compute net injections per bus
P_inj = gen_to_bus @ Pg - Pd

# Exclude slack bus from PTDF
indices = [i for i in range(num_buses) if i != id_to_index[slack]]
P_inj_reduced = P_inj[indices]

# Objective: Minimize generation cost
objective = cp.Minimize(cp.sum(cp.multiply(cost_coeff, Pg)))

# Constraints
power_balance = cp.sum(Pg) == np.sum(Pd)  # Total power balance
constraints = [
    power_balance,
    Pg_min <= Pg ,
    Pg <= Pg_max,
    -branch_data[:, 5] <= PTDF @ P_inj_reduced,
    PTDF @ P_inj_reduced <= branch_data[:, 5]
]

# Solve the optimization problem
prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.MOSEK, verbose=True)
if prob.status == cp.OPTIMAL:
    print(f"status: {prob.status}")
    print(f"Optimal value: {prob.value}")
    print(f"Optimal generation: {Pg.value}")
else :
    print("Problem is not optimal. Status: ", prob.status)
    exit()



