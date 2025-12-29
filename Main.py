import pulp
import networkx as nx
import matplotlib.pyplot as plt
import os


def check_solution_integrity(lp_model, tolerance=1e-6):
    # Check solver status first
    status_str = pulp.LpStatus[lp_model.status]
    if status_str != 'Optimal':
        print(f"Model status is not optimal: {status_str}")
        return False

    # Check constraints feasibility
    for name, constraint in lp_model.constraints.items():
        val = pulp.value(constraint)  # This is the normalized lhs - rhs
        sense = constraint.sense

        # Based on the sense:
        if sense == pulp.LpConstraintLE:
            # Constraint is: expression <= 0
            # val should be <= 0
            if val > tolerance:
                print(f"Constraint {name} is violated: Value={val}, must be ≤ 0.")
                return False
        elif sense == pulp.LpConstraintGE:
            # Constraint is: expression >= 0
            # val should be >= 0
            if val < -tolerance:
                print(f"Constraint {name} is violated: Value={val}, must be ≥ 0.")
                return False
        elif sense == pulp.LpConstraintEQ:
            # Constraint is: expression == 0
            # val should be ~0 within tolerance
            if abs(val) > tolerance:
                print(f"Constraint {name} is violated: Value={val}, must be ≈ 0.")
                return False

    # Check non-negativity/bounds for variables
    for v in lp_model.variables():
        val = v.varValue
        if val is None or val < -tolerance:
            print(f"Variable {v.name} has invalid value: {val}")
            return False

    print("All constraints and variable bounds appear to be satisfied.")
    return True


def draw_network_flows(X_S_T1, X_T1_T2, X_T2_D,
                       no_of_supply_nodes,
                       no_of_first_layer_transshipment_nodes,
                       no_of_second_layer_transshipment_nodes,
                       no_of_demand_nodes,
                       flow_tolerance=1e-6, title="", filename=None):
    """
    Draw a multi-layer transshipment network using the provided flow variables.
    This version arranges layers horizontally, left to right,
    and includes tweaks to reduce overlapping labels.
    """

    supply_nodes = range(no_of_supply_nodes)
    first_layer_nodes = range(no_of_first_layer_transshipment_nodes)
    second_layer_nodes = range(no_of_second_layer_transshipment_nodes)
    demand_nodes = range(no_of_demand_nodes)

    # Extract positive flows
    flow_S_T1 = {(i, j): X_S_T1[i][j].varValue for i in supply_nodes for j in first_layer_nodes
                 if X_S_T1[i][j].varValue > flow_tolerance}
    flow_T1_T2 = {(j, k): X_T1_T2[j][k].varValue for j in first_layer_nodes for k in second_layer_nodes
                  if X_T1_T2[j][k].varValue > flow_tolerance}
    flow_T2_D = {(k, l): X_T2_D[k][l].varValue for k in second_layer_nodes for l in demand_nodes
                 if X_T2_D[k][l].varValue > flow_tolerance}

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes with layer attribute
    for i in supply_nodes:
        G.add_node(f"S{i}", layer=0)
    for j in first_layer_nodes:
        G.add_node(f"T1_{j}", layer=1)
    for k in second_layer_nodes:
        G.add_node(f"T2_{k}", layer=2)
    for l in demand_nodes:
        G.add_node(f"D{l}", layer=3)

    # Add edges for positive flows
    for (i, j), val in flow_S_T1.items():
        G.add_edge(f"S{i}", f"T1_{j}", flow=val)

    for (j, k), val in flow_T1_T2.items():
        G.add_edge(f"T1_{j}", f"T2_{k}", flow=val)

    for (k, l), val in flow_T2_D.items():
        G.add_edge(f"T2_{k}", f"D{l}", flow=val)

    # Determine position for each layer of nodes
    layer_nodes = {
        0: [n for n in G.nodes() if G.nodes[n]['layer'] == 0],
        1: [n for n in G.nodes() if G.nodes[n]['layer'] == 1],
        2: [n for n in G.nodes() if G.nodes[n]['layer'] == 2],
        3: [n for n in G.nodes() if G.nodes[n]['layer'] == 3]
    }

    pos = {}
    x_spacing = 3.0
    y_spacing = 1.5  # Increased spacing to reduce overlap

    # Positioning each layer horizontally, nodes stacked vertically
    for layer, nodes_in_layer in layer_nodes.items():
        for idx, node in enumerate(nodes_in_layer):
            # layer determines x-position, idx determines y-position
            pos[node] = (layer * x_spacing, -idx * y_spacing)

    # Increase figure size for less overlap
    plt.figure(figsize=(12, 6))

    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1200, edgecolors='black')
    nx.draw_networkx_labels(G, pos, font_size=9, font_color='black')  # Slightly smaller font

    # Prepare edge properties
    edges = G.edges(data=True)
    edge_labels = {(u, v): f"{d['flow']:.1f}" for u, v, d in edges}
    # Scale widths by flow
    edge_widths = [max(0.5, d['flow'] / 50.0) for _, _, d in edges]

    # Draw edges
    nx.draw_networkx_edges(G, pos, width=edge_widths, arrowstyle='-|>', arrowsize=20)

    # Try adjusting label_pos to reduce overlapping labels
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, label_pos=0.3,
                                 bbox=dict(facecolor='white', edgecolor='none'))

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    # if output folder doesn't exist, create it
    if not os.path.exists("output"):
        os.makedirs("output")

    # Save plot if filename is provided
    if filename:
        plt.savefig(filename)
    plt.show()


def run_transshipment_problem(c_S_T1, c_T1_T2, c_T2_D, supply_capacities, demand_quantities, title="", filename=None):
    """Runs the transshipment problem given the input data and prints results."""
    no_of_supply_nodes = len(c_S_T1)
    no_of_first_layer_transshipment_nodes = len(c_S_T1[0])
    no_of_second_layer_transshipment_nodes = len(c_T1_T2[0])
    no_of_demand_nodes = len(c_T2_D[0])

    # Initialize the model
    model = pulp.LpProblem("Multiple_Layer_Transshipment", pulp.LpMinimize)

    # Define decision variables
    X_S_T1 = pulp.LpVariable.dicts("X_S_T1",
                                   (range(no_of_supply_nodes), range(no_of_first_layer_transshipment_nodes)),
                                   lowBound=0, cat=pulp.LpContinuous)

    X_T1_T2 = pulp.LpVariable.dicts("X_T1_T2",
                                    (range(no_of_first_layer_transshipment_nodes),
                                     range(no_of_second_layer_transshipment_nodes)),
                                    lowBound=0, cat=pulp.LpContinuous)

    X_T2_D = pulp.LpVariable.dicts("X_T2_D",
                                   (range(no_of_second_layer_transshipment_nodes), range(no_of_demand_nodes)),
                                   lowBound=0, cat=pulp.LpContinuous)

    # Objective Function
    model += pulp.lpSum([c_S_T1[i][j] * X_S_T1[i][j] for i in range(no_of_supply_nodes) for j in
                         range(no_of_first_layer_transshipment_nodes)]) \
             + pulp.lpSum([c_T1_T2[j][k] * X_T1_T2[j][k] for j in range(no_of_first_layer_transshipment_nodes) for k in
                           range(no_of_second_layer_transshipment_nodes)]) \
             + pulp.lpSum([c_T2_D[k][l] * X_T2_D[k][l] for k in range(no_of_second_layer_transshipment_nodes) for l in
                           range(no_of_demand_nodes)]), "Total_Cost"

    # Supply constraints
    for i in range(no_of_supply_nodes):
        model += pulp.lpSum([X_S_T1[i][j] for j in range(no_of_first_layer_transshipment_nodes)]) <= supply_capacities[
            i], f"SupplyCap_{i}"

    # Flow conservation at first-layer nodes
    for j in range(no_of_first_layer_transshipment_nodes):
        model += pulp.lpSum([X_S_T1[i][j] for i in range(no_of_supply_nodes)]) == pulp.lpSum(
            [X_T1_T2[j][k] for k in range(no_of_second_layer_transshipment_nodes)]), f"FlowFirstLayer_{j}"

    # Flow conservation at second-layer nodes
    for k in range(no_of_second_layer_transshipment_nodes):
        model += pulp.lpSum([X_T1_T2[j][k] for j in range(no_of_first_layer_transshipment_nodes)]) == pulp.lpSum(
            [X_T2_D[k][l] for l in range(no_of_demand_nodes)]), f"FlowSecondLayer_{k}"

    # Demand constraints
    for l in range(no_of_demand_nodes):
        model += pulp.lpSum([X_T2_D[k][l] for k in range(no_of_second_layer_transshipment_nodes)]) == demand_quantities[
            l], f"Demand_{l}"

    # Solve the model
    model.solve(pulp.PULP_CBC_CMD(msg=0))

    # Run the verification check
    if check_solution_integrity(model):
        # If all checks passed, print the results
        print("Status:", pulp.LpStatus[model.status])
        print("Optimal Total Cost:", pulp.value(model.objective))
        print("\nDecision Variables:")
        for v in model.variables():
            if v.varValue > 1e-6:  # to filter out zero or near-zero flow
                print(v.name, "=", v.varValue)

        # Call the drawing function
        draw_network_flows(X_S_T1, X_T1_T2, X_T2_D,
                           no_of_supply_nodes,
                           no_of_first_layer_transshipment_nodes,
                           no_of_second_layer_transshipment_nodes,
                           no_of_demand_nodes,
                           title=title, filename=filename)
    else:
        print("Solution verification failed.")


# Define datasets for each question
questions = {
    "Q1": {
        "data": {
            "c_S_T1": [
                [25, 17, 21, 13],
                [26, 24, 14, 28],
                [18, 12, 16, 27],
                [26, 11, 10, 21]
            ],
            "c_T1_T2": [
                [55, 34, 57, 58],
                [47, 52, 63, 59],
                [69, 48, 30, 48],
                [67, 46, 60, 64]
            ],
            "c_T2_D": [
                [15, 15, 19, 10, 19],
                [10, 16, 14, 16, 11],
                [17, 10, 18, 14, 15],
                [11, 12, 19, 18, 10]
            ],
            "supply_capacities": [335, 327, 337, 247],
            "demand_quantities": [237, 298, 192, 158, 313]
        },
        "title": "Optimal Flows: Question 1",
        "filename": "output/Q1_optimal_flows.png"
    },
}

# Define specific changes for each sub-question in Question 2 and beyond
additional_questions = {
    "Q2_a": {
        "data": {
            **questions["Q1"]["data"],  # Start with baseline data
            "supply_capacities": [336, 327, 337, 247],  # Increase capacity of supply node 0 by 1
        },
        "title": "Optimal Flows: Question 2.a",
        "filename": "output/Q2_a_optimal_flows.png"
    },
    "Q2_b": {
        "data": {
            **questions["Q1"]["data"],  # Start with baseline data
            "supply_capacities": [335, 328, 337, 247],  # Increase capacity of supply node 1 by 1
        },
        "title": "Optimal Flows: Question 2.b",
        "filename": "output/Q2_b_optimal_flows.png"
    },
    "Q2_c": {
        "data": {
            **questions["Q1"]["data"],  # Start with baseline data
            "demand_quantities": [237, 299, 192, 158, 313],  # Increase demand for demand node 1 by 1
        },
        "title": "Optimal Flows: Question 2.c",
        "filename": "output/Q2_c_optimal_flows.png"
    },
    "Q2_d": {
        "data": {
            **questions["Q1"]["data"],  # Start with baseline data
            "c_S_T1": [
                [25, 17, 20, 13],  # Decrease cost from supply node 0 to T1_2 by 1
                [26, 24, 14, 28],
                [18, 12, 16, 27],
                [26, 11, 10, 21],
            ],
        },
        "title": "Optimal Flows: Question 2.d",
        "filename": "output/Q2_d_optimal_flows.png"
    },
    "Q2_e": {
        "data": {
            **questions["Q1"]["data"],  # Start with baseline data
            "c_S_T1": [
                [24, 17, 21, 13],  # Decrease cost from supply node 0 to T1_0 by 1
                [26, 24, 14, 28],
                [18, 12, 16, 27],
                [26, 11, 10, 21],
            ],
        },
        "title": "Optimal Flows: Question 2.e",
        "filename": "output/Q2_e_optimal_flows.png"
    },
    "Q2_f": {
        "data": {
            **questions["Q1"]["data"],  # Start with baseline data
            "c_S_T1": [
                [18, 17, 21, 13],  # Decrease cost from supply node 0 to T1_0 by 7
                [26, 24, 14, 28],
                [18, 12, 16, 27],
                [26, 11, 10, 21],
            ],
        },
        "title": "Optimal Flows: Question 2.f",
        "filename": "output/Q2_f_optimal_flows.png"
    },
}

# Merge all questions into one
all_questions = {**questions, **additional_questions}

# Run all questions and save results
for key, question in all_questions.items():
    print("\n---------------------------------------")
    print(f"Running {key}...")
    print(f"Title: {question['title']}")
    print("---------------------------------------")
    run_transshipment_problem(**question["data"], title=question["title"], filename=question["filename"])
    print(f"### Completed {key} ###\n")
    print("-" * 50)

# Separate section for Question 3
print("\n### Question 3: Blocking Supply Node 0 to T1_2 ###\n")
print("Hypothesis: Blocking this route will likely increase the total cost due to reduced flexibility in routing.")

# Modify the data to simulate blocking the route
q3_data = {
    **questions["Q1"]["data"],  # Base data
    "c_S_T1": [
        [25, 17, 10 ** 6, 13],  # Set a high cost to block the route from supply node 0 to T1_2
        [26, 24, 14, 28],
        [18, 12, 16, 27],
        [26, 11, 10, 21]
    ]
}

# Run and solve the model for Question 3
run_transshipment_problem(
    **q3_data,
    title="Optimal Flows: Question 3",
    filename="output/Q3_blocking_route.png"
)

# Separate section for Question 4
print("\n### Question 4 Modified Approach ###\n")
print("We will incrementally increase the demand of retailer 0 and see at which point the solution becomes infeasible.")
base_data = {
    "c_S_T1": [
        [25, 17, 21, 13],
        [26, 24, 14, 28],
        [18, 12, 16, 27],
        [26, 11, 10, 21]
    ],
    "c_T1_T2": [
        [55, 34, 57, 58],
        [47, 52, 63, 59],
        [69, 48, 30, 48],
        [67, 46, 60, 64]
    ],
    "c_T2_D": [
        [15, 15, 19, 10, 19],
        [10, 16, 14, 16, 11],
        [17, 10, 18, 14, 15],
        [11, 12, 19, 18, 10]
    ],
    "supply_capacities": [335, 327, 337, 247],
    "demand_quantities": [237, 298, 192, 158, 313]
}

start_demand = base_data["demand_quantities"][0]
increments = [start_demand + i for i in range(0, 81, 10)]  # from original 237 up to 317 in increments of 10

for new_demand in increments:
    q4_data = {
        **base_data,
        "demand_quantities": [new_demand, 298, 192, 158, 313]
    }
    print(f"\nTrying demand of retailer 0 = {new_demand}")
    run_transshipment_problem(**q4_data, title=f"Q4 Demand at Retailer 0 = {new_demand}", filename=None)
    print("-" * 50)

print("From this incremental approach, we can identify the highest feasible demand for retailer 0.")
print(
    "Once we hit an infeasible solution, the previous step indicates the limit of what the current network configuration and supplies can support.")