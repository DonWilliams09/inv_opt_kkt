import numpy as np
import cvxpy as cp
import re
import json
import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


# Function to extract data from MATLAB .m file
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

# Function to extract huge data from MATLAB .m file
import numpy as np
import re

def parse_big_m_file(filepath):
    """Robustly parses large MATLAB .m power flow case files from MATPOWER format."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    def extract_big_data(section_name):
        start = False
        data = []
        expected_cols = None

        for line in lines:
            stripped = line.strip()
            if not start and f"mpc.{section_name} = [" in stripped:
                start = True
                continue
            if start:
                if '];' in stripped:
                    break
                if stripped.startswith('%') or stripped == '':
                    continue
                numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", stripped)
                if numbers:
                    row = [float(num) for num in numbers]
                    if expected_cols is None:
                        expected_cols = len(row)
                    if len(row) == expected_cols:
                        data.append(row)
                    else:
                        print(f"[WARN] Skipping malformed row in section {section_name}: {row}")

        if not data:
            raise ValueError(f"Section {section_name} not found or empty.")

        return np.array(data)

    # Parse all sections
    bus_data = extract_big_data("bus")
    gen_data = extract_big_data("gen")
    branch_data = extract_big_data("branch")
    gencost_data = extract_big_data("gencost")

    return bus_data, gen_data, branch_data, gencost_data



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


# Function to plot power network
def plot_power_network(power_flows, lmp, branch_data, title="IEEE 14-Bus Power Flow"):
    # Define bus coordinates (IEEE 14-bus layout)
    bus_coords = {
        0: (0, 0),    1: (1, 1),   2: (2, 2),
        3: (3, 1.2),  4: (2, 0),   5: (2.5, -0.5),
        6: (0.5, -1), 7: (0, -2),  8: (1.5, -2.5),
        9: (2.5, -2.5), 10: (3.5, -2), 11: (4, -1),
        12: (4.5, 0), 13: (5, 2.5)
    }

    num_buses = len(bus_coords)

    # Create graph
    G = nx.Graph()
    for i in range(num_buses):
        G.add_node(i, pos=bus_coords[i])

    for i, (f, t) in enumerate(zip(branch_data[:, 0].astype(int) - 1,
                                   branch_data[:, 1].astype(int) - 1)):
        G.add_edge(f, t, index=i)

    pos = nx.get_node_attributes(G, 'pos')

    # Edge colors and widths
    edge_colors = []
    edge_widths = []
    for (u, v, d) in G.edges(data=True):
        idx = d['index']
        pf = power_flows[idx]
        limit = branch_data[idx, 5]
        color = 'red' if abs(pf) >= 0.98 * limit else 'gray'
        edge_colors.append(color)
        edge_widths.append(2 + 6 * abs(pf) / np.max(np.abs(power_flows)))

    # Plot
    plt.figure(figsize=(10, 6))
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths)
    nx.draw_networkx_labels(G, pos, font_size=10)

    # Annotate LMPs
    for i in range(num_buses):
        x, y = pos[i]
        plt.text(x, y + 0.25, f"LMP: {lmp[i]:.2f}", fontsize=8, ha='center')

    # Annotate line flows
    for (u, v, d) in G.edges(data=True):
        idx = d['index']
        mid_x = (pos[u][0] + pos[v][0]) / 2
        mid_y = (pos[u][1] + pos[v][1]) / 2
        pf = power_flows[idx]
        plt.text(mid_x, mid_y, f"{pf:.1f}", fontsize=7, color='black', ha='center')

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def plot_cost_comparison(true_costs, inferred_costs, title="True vs Inferred Cost Coefficients"):
    """
    Plots a bar chart comparing true and inferred generator cost coefficients.

    Parameters:
    -----------
    true_costs : np.ndarray
        Array of true cost coefficients (e.g., from gencost_data[:, 5]).
    inferred_costs : np.ndarray
        Array of inferred cost coefficients (e.g., from c.value after inverse optimization).
    title : str
        Title of the plot.
    """
    assert len(true_costs) == len(inferred_costs), "Input arrays must be the same length."

    num_generators = len(true_costs)
    bar_width = 0.35
    index = np.arange(num_generators)

    plt.figure(figsize=(10, 6))
    plt.bar(index, true_costs, bar_width, label='True Cost Coefficients')
    plt.bar(index + bar_width, inferred_costs, bar_width, label='Inferred Cost Coefficients')

    plt.xlabel('Generator Index')
    plt.ylabel('Linear Cost Coefficient')
    plt.title(title)
    plt.xticks(index + bar_width / 2, [f'Gen {i+1}' for i in range(num_generators)])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
