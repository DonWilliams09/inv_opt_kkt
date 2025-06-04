import numpy as np
import cvxpy as cp
import re

np.set_printoptions(precision=3, suppress=True, floatmode='fixed')

def parse_m_file(filepath):
    """Parses MATLAB .m power flow case file manually."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    def extract_data(section_name):
        """Extracts numerical matrix data from a given section."""
        start = None
        for i, line in enumerate(lines):
            if line.strip().startswith(f"mpc.{section_name} = ["):
                start = i + 1
                break
        
        if start is None:
            raise ValueError(f"Section {section_name} not found in file.")

        data = []
        for line in lines[start:]:
            if '];' in line:
                break
            numbers = [float(num) for num in re.findall(r"[-+]?\d*\.\d+|\d+", line)]
            data.append(numbers)
        
        return np.array(data)

    # Extract sections
    bus_data = extract_data("bus")
    gen_data = extract_data("gen")
    branch_data = extract_data("branch")
    gencost_data = extract_data("gencost")

    return bus_data, gen_data, branch_data, gencost_data

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
cost_coeff = gencost_data[:, 5]  # Linear cost coefficient

# Function to compute PTDF
def compute_PTDF(branch_data, bus_data, slack_bus=0):
    """ Compute PTDF matrix given branch and bus data """
    num_buses = bus_data.shape[0]
    num_lines = branch_data.shape[0]

    # Admittance matrix (B)
    B = np.zeros((num_buses, num_buses))
    for f, t, x in zip(branch_data[:, 0].astype(int) - 1,
                       branch_data[:, 1].astype(int) - 1,
                       branch_data[:, 3]):
        B[f, t] = -1 / x
        B[t, f] = -1 / x
    for i in range(num_buses):
        B[i, i] = -np.sum(B[i, :])

    # Remove slack bus row/col
    B_reduced = np.delete(np.delete(B, slack_bus, axis=0), slack_bus, axis=1)
    B_inv = np.linalg.inv(B_reduced)

    # Compute B_line
    B_line = np.zeros((num_lines, num_buses))
    for idx, (f, t, x) in enumerate(zip(branch_data[:, 0].astype(int) - 1,
                                        branch_data[:, 1].astype(int) - 1,
                                        branch_data[:, 3])):
        B_line[idx, f] = 1 / x
        B_line[idx, t] = -1 / x

    # Remove slack column
    B_line_reduced = B_line[:, 1:]

    # Compute PTDF
    PTDF = B_line_reduced @ B_inv

    return PTDF

# Compute PTDF
PTDF = compute_PTDF(branch_data, bus_data)

# Define optimization variables
Pg = cp.Variable(num_generators)

# Create generator-to-bus incidence matrix
gen_to_bus = np.zeros((num_buses, num_generators))
for i, gen_bus in enumerate(gen_data[:, 0].astype(int) - 1):
    gen_to_bus[gen_bus, i] = 1

# Compute net injections per bus
P_inj = gen_to_bus @ Pg - Pd
P_inj_reduced = P_inj[1:]  # Remove slack bus

# Objective: Minimize generation cost
objective = cp.Minimize(cp.sum(cp.multiply(cost_coeff, Pg)))

# Constraints
constraints = [
    cp.sum(Pg) == np.sum(Pd), # Power balance
    Pg >= Pg_min,
    Pg <= Pg_max,
    -branch_data[:, 5] <= PTDF @ P_inj_reduced,
    PTDF @ P_inj_reduced <= branch_data[:, 5]
]

# Solve the optimization problem
prob = cp.Problem(objective, constraints)
# Solve the optimization problem
try:
    prob.solve(solver=cp.MOSEK)
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print(f"Le problème n'a pas été résolu avec succès. Statut : {prob.status}")
        exit()
except Exception as e:
    print("Erreur lors de la résolution :", e)
    exit()

# --- Post-processing ---
if Pg.value is not None:
    print("Optimal Generation Dispatch (MW):")
    for i, val in enumerate(Pg.value):
        print(f"Generator {i+1}: {val:.2f} MW")

    print(f"\nOptimal Cost: {prob.value:.2f} $")
else:
    print("Aucune solution trouvée.")

# Optimal generation
print("Optimal Generation Dispatch (MW):")
for i, val in enumerate(Pg.value):
    print(f"Generator {i+1}: {val:.2f} MW")

print(f"\nOptimal Cost: {prob.value:.2f} $")

# Power flows
P_inj_opt = gen_to_bus @ Pg.value - Pd
P_inj_reduced_opt = P_inj_opt[1:]
power_flows = PTDF @ P_inj_reduced_opt

print("\n--- Power Flows on Branches (MW) ---")
for i, pf in enumerate(power_flows):
    print(f"Line {i+1}: {pf:.2f} MW (Limit: ±{branch_data[i, 5]} MW)")

# Dual values of flow constraints
lower_flow_duals = constraints[3].dual_value
upper_flow_duals = constraints[4].dual_value

print("\n--- Dual Values of Flow Constraints ---")
for i in range(num_lines):
    print(f"Line {i+1}: Lower Dual = {lower_flow_duals[i]:.4f}, Upper Dual = {upper_flow_duals[i]:.4f}")

# LMPs (Locational Marginal Prices)
lam = np.zeros(num_buses)
lam[1:] = constraints[0].dual_value  # LMPs at all buses except slack
# Optional: you can set slack LMP arbitrarily (e.g., 0 or average of neighbors)

print("\n--- Locational Marginal Prices (LMPs) ---")
for i in range(num_buses):
    print(f"Bus {i+1}: LMP = {lam[i]:.4f} $/MWh")
