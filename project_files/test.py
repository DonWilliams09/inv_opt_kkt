import numpy as np
import cvxpy as cp
#from utils import parse_m_file, compute_PTDF
from oct2py import Oct2Py

np.set_printoptions(precision=3, suppress=True, floatmode='fixed')


# Start Octave session
oc = Oct2Py()

# Load the MATPOWER case file
oc.eval("addpath('/Users/don_williams09/Downloads/Bi_Level_Opt')")  # Adjust path to the folder containing the file
mpc = oc.case300()

# Extract data
bus_data = mpc['bus']
gen_data = mpc['gen']
branch_data = mpc['branch']
gencost_data = mpc['gencost']

# Show dimensions or samples
print(f"Bus data shape : {bus_data.shape} ")
print(f" Generator data shape : {gen_data.shape}")
print(f" Branch data shape : {branch_data.shape} ")
print(f" Gencost data shape : {gencost_data.shape}")

num_buses = bus_data.shape[0]
num_generators = gen_data.shape[0]
num_lines = branch_data.shape[0]
num_branches = branch_data.shape[0]

# Data Extraction
Pd_base = bus_data[:, 2]
Pg_min = gen_data[:, 9]
Pg_max = gen_data[:, 8]
# Pg_max[0] = 400
# Pg_max[1] = 100
cost_coeff_true = gencost_data[:, 5]

branch_data_congested = branch_data.copy()
#branch_data_congested[:, 5] *= 0.7  # reduce limits by 30%

# PTDF = compute_PTDF(branch_data, bus_data)
import numpy as np

def compute_PTDF(branch_data, bus_data, slack_bus=0):
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
    slack_index = id_to_index[slack_bus] if slack_bus in id_to_index else 0
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

    return PTDF

PTDF = compute_PTDF(branch_data_congested, bus_data)


# Generator incidence matrix
bus_ids = bus_data[:, 0].astype(int)
id_to_index = {bus_id: idx for idx, bus_id in enumerate(bus_ids)}
gen_to_bus = np.zeros((num_buses, num_generators))
for i, gen_bus_id in enumerate(gen_data[:, 0].astype(int)):
    bus_index = id_to_index[gen_bus_id]
    gen_to_bus[bus_index, i] = 1

# Storage for scenario data
Pg_scenarios = []
lambda_slack_scenarios = []
nu_max_scenarios = []
nu_min_scenarios = []
mu_min_scenarios = []
mu_max_scenarios = []
Pd_scenarios = []

perturbation_scale = 0.3
num_scenarios = 2
scenario_count = 0
np.random.seed(42)

# while scenario_count < num_scenarios:
#Pd_perturbed = Pd_base * (1 + np.random.uniform(-perturbation_scale, perturbation_scale, size=num_buses))
Pd_perturbed = Pd_base.copy()
#Pd_perturbed = np.random.randint(0, 100, size=Pd_base.shape) + np.random.randint(0, 100, size=Pd_base.shape)/100
#Pd_perturbed[0] = 0  # slack bus
Pd_scenarios.append(Pd_perturbed)

Pg = cp.Variable(num_generators)
P_inj = gen_to_bus @ Pg - Pd_perturbed
P_inj_reduced = P_inj[1:]

constraints = [
    cp.sum(Pg) == np.sum(Pd_perturbed),
    Pg >= Pg_min,
    Pg <= Pg_max,
    -branch_data_congested[:, 5] <= PTDF @ P_inj_reduced,
    PTDF @ P_inj_reduced <= branch_data_congested[:, 5]
]

objective = cp.Minimize(cp.sum(cp.multiply(cost_coeff_true, Pg)))
prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.MOSEK, verbose=False)

if prob.status == 'optimal':
    scenario_count += 1
    Pg_scenarios.append(Pg.value)
    Pd_scenarios.append(Pd_perturbed)
    lambda_slack_scenarios.append(constraints[0].dual_value)
    nu_min_scenarios.append(constraints[3].dual_value)
    nu_max_scenarios.append(constraints[4].dual_value)
    mu_min_scenarios.append(constraints[1].dual_value)
    mu_max_scenarios.append(constraints[2].dual_value)
    print(f"Scenario {scenario_count}/{num_scenarios} generated.")
else:
    print(f"Scenario generation failed with status: {prob.status}")

Pg_scenarios = np.array(Pg_scenarios)
lambda_slack_scenarios = np.array(lambda_slack_scenarios)
nu_max_scenarios = np.array(nu_max_scenarios)
nu_min_scenarios = np.array(nu_min_scenarios)
mu_min_scenarios = np.array(mu_min_scenarios)
mu_max_scenarios = np.array(mu_max_scenarios)

print("Pg_scenarios: ", Pg_scenarios)

