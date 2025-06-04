import numpy as np

# --- Build B matrix as you did before ---
B = np.zeros((5, 5))
B[0,1] = -1/0.0064; B[1,0] = B[0,1]
B[0,2] = -1/0.0281; B[2,0] = B[0,2]
B[2,3] = -1/0.0297; B[3,2] = B[2,3]
B[3,4] = -1/0.0297; B[4,3] = B[3,4]
B[1,4] = -1/0.0304; B[4,1] = B[1,4]

for i in range(5):
    B[i,i] = -sum(B[i,:])

# --- Reduce B by removing slack bus (bus 0) ---
B_reduced = B[1:, 1:]  # B̃_bus

# --- Invert reduced matrix ---
B_inv = np.linalg.inv(B_reduced)

# --- Define lines (from_bus, to_bus, reactance) ---
lines = [
    (0, 1, 0.0064),
    (0, 2, 0.0281),
    (2, 3, 0.0297),
    (3, 4, 0.0297),
    (1, 4, 0.0304)
]

# --- Build B_line matrix ---
num_lines = len(lines)
num_buses = 5
B_line = np.zeros((num_lines, num_buses))

for idx, (f, t, x) in enumerate(lines):
    print(f"idx = {idx}, f = {f}, t = {t} x = {x}")
    B_line[idx, f] = 1 / x
    B_line[idx, t] = -1 / x
    print(B_line)

# --- Remove slack column (bus 0) to match reduced theta dimension ---
B_line_reduced = B_line[:, 1:]

# --- PTDF = B_line * inv(B̃_bus) ---
PTDF = B_line_reduced @ B_inv

# --- Display with nice formatting ---
np.set_printoptions(precision=3, suppress=True, floatmode='fixed')
print("PTDF matrix =\n", PTDF.T)
