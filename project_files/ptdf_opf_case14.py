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

# print(bus_data)
# print(branch_data)
# print(gen_data)
# print(gencost_data)

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
    for f, t, x in zip(branch_data[:, 0].astype(int) - 1, #fbus --> start line idx
                        branch_data[:, 1].astype(int) - 1, #tbus --> end line idx
                        branch_data[:, 3]): # Reactance X of the line
        B[f, t] = -1 / x
        B[t, f] = -1 / x
    for i in range(num_buses):
        B[i, i] = -np.sum(B[i, :])

    # Remove slack bus row/col to compute B_reduced
    B_reduced = np.delete(np.delete(B, slack_bus, axis=0), slack_bus, axis=1)
    #B_reduced = B.copy()
    #B_reduced = B[1:, 1:]
    
    # Compute the inverse of the reduced B matrix
    B_inv = np.linalg.inv(B_reduced)
    
    # Compute the B_line matrix
    
    B_line = np.zeros((num_lines, num_buses))
    for idx, (f, t, x) in enumerate(zip(branch_data[:, 0].astype(int) - 1, 
                                         branch_data[:, 1].astype(int) - 1, 
                                         branch_data[:, 3])):
        B_line[idx, f] = 1 / x
        B_line[idx, t] = -1 / x
    
    # Remove slack column (bus 0)
    B_line_reduced = B_line[:, 1:]
    
    # Compute PTDF
    PTDF = B_line_reduced@B_inv
    
    
    return PTDF

# Compute PTDF
PTDF = compute_PTDF(branch_data, bus_data)
#print("PTDF matrix =\n", PTDF)

# Define optimization variables
Pg = cp.Variable(num_generators)

# Create generator-to-bus incidence matrix
gen_to_bus = np.zeros((num_buses, num_generators))
for i, gen_bus in enumerate(gen_data[:, 0].astype(int) - 1):  # Generator bus indices
    gen_to_bus[gen_bus, i] = 1

# Compute net injections per bus
P_inj = gen_to_bus @ Pg - Pd
P_inj_reduced = P_inj[1:]  # Suppression du slack bus (indice 0)

# Objective: Minimize generation cost
objective = cp.Minimize(cp.sum(cp.multiply(cost_coeff, Pg)))

# Constraints
constraints = [
    cp.sum(Pg) == np.sum(Pd),  # Power balance
    Pg >= Pg_min,
    Pg <= Pg_max,
    -branch_data[:, 5] <= PTDF @ P_inj_reduced,  # Line limits (PTDF formulation)
    PTDF @ P_inj_reduced <= branch_data[:, 5]
]

# Solve the optimization problem
prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.MOSEK)
#prob.get_problem_data(cp.MOSEK)

# Print results
print("Optimal Generation Dispatch (MW):", Pg.value)
print("Optimal Cost:", prob.value)

if prob.status == cp.OPTIMAL:
    print("Optimal power generation:", Pg.value)
    print("Optimal cost:", prob.value)

    # Calculate power flow on each line using PTDF and the optimal generation and demand values
    Pg_value = Pg.value
    print(Pg_value)
    # power_flows = T @ (Pg_value - Pd)
    P_inj = gen_to_bus @ Pg_value - Pd
    P_inj_reduced = P_inj[1:]

    power_flows = PTDF @ P_inj_reduced

    # Display the power flow on each line (1-2, 1-3, 2-3)
    line_1_2 = power_flows[0]
    line_1_3 = power_flows[1]
    line_2_3 = power_flows[2]
    #print(Pg_value[:,J-1])
    print(f"Power flow on line 1-2: {line_1_2} MW, limit: {branch_data[0, 5]} MW, dual: {constraints[3].dual_value}")
    print(f"Power flow on line 1-3: {line_1_3} MW, limit: {branch_data[1, 5]} MW, dual: {constraints[3].dual_value}")
    print(f"Power flow on line 2-3: {line_2_3} MW, limit: {branch_data[2, 5]} MW, dual: {constraints[3].dual_value}")
else:
    print("Optimization did not converge. Status:", prob.status)