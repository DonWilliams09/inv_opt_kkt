import numpy as np
import cvxpy as cp
import mosek
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
n = 3
J = 744

# Define the parameters for a 3-bus system
Pg = cp.Variable((3,J))  # Power generation variables
#Pg = cp.Variable(3)
Pd = np.array([np.array([random.uniform(0, 100) for _ in range(J)]), np.array([random.uniform(0, 100) for _ in range(J)]), np.array([random.uniform(0, 200) for _ in range(J)])])  # Power demand at each bus
#Pd = np.array([0,100,200])
T = np.array([[1/3, -1/3, 0],
              [2/3, 1/3, 0],
              [1/3, 2/3, 0]])  # Updated PTDF matrix
Tmax = np.array([50.0, 100.0, 100.0])  # Physical limits for the power flow on each line
Pmin = np.array([0.0, 0.0, 0.0])  # Minimum power generation
Pmax = np.array([1000, 1000, 1000])  # Maximum power generation

# Define the cost function c(Pg)
c = np.array([40.0, 80.0, 140.0])  # Cost coefficients for linear cost function
# cost_function = c@Pg
for j in range(J):
  cost_function = c @  Pg[:,j]

# Define the constraints
constraints = []
# constraints += [cp.sum(Pg - Pd) == 0]
# constraints += [-Tmax <= T @ (Pg - Pd)]
# constraints += [ T @ (Pg - Pd) <= Tmax]
# constraints +=[Pmin <= Pg,  Pg<= Pmax]  # Maximum power generation
for j in range(J):
  constraints += [cp.sum(Pg[:,j] - Pd[:,j]) == 0]
  constraints += [-Tmax <= T @ (Pg[:,j] - Pd[:,j])]
  constraints += [ T @ (Pg[:,j] - Pd[:,j]) <= Tmax]
  constraints +=[Pmin <= Pg[:,j],  Pg[:,j] <= Pmax]  # Maximum power generation


# Define the optimization problem
problem = cp.Problem(cp.Minimize(cost_function), constraints)

# Solve the problem
problem.solve(verbose = True)
print(Pd)
# Check if the solution is feasible
if problem.status == cp.OPTIMAL:
    print("Optimal power generation:", Pg.value)
    print("Optimal cost:", problem.value)

    # Calculate power flow on each line using PTDF and the optimal generation and demand values
    Pg_value = Pg.value
    print(Pg_value)
    # power_flows = T @ (Pg_value - Pd)
    for j in range(J):
      power_flows = T @ (Pg_value[:,j] - Pd[:,j])

    # Display the power flow on each line (1-2, 1-3, 2-3)
    line_1_2 = power_flows[0]
    line_1_3 = power_flows[1]
    line_2_3 = power_flows[2]
    #print(Pg_value[:,J-1])
    print(f"Power flow on line 1-2: {line_1_2} MW")
    print(f"Power flow on line 1-3: {line_1_3} MW")
    print(f"Power flow on line 2-3: {line_2_3} MW")
else:
    print("Optimization did not converge. Status:", problem.status)
