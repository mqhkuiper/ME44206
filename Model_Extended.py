from gurobipy import *
import numpy as np
import math
import matplotlib.pyplot as plt

# ============================================MODEL DATA============================================

# Load data from file
with open("data_small_multiTW.txt", "r") as f:
    data = f.readlines()

VRP = []
for line in data:
    words = line.split()
    words = [int(i) for i in words]
    VRP.append(words)

VRP = np.array(VRP)

# Parsing the data structure
N = VRP[:, 0]  # Nodes (including depot)
C = VRP[1:-1, 0]  # Customers/Locations
V = [0, 1]  # Vehicles
Q = 130  # Vehicle capacity
d = VRP[:, 3]  # Demand
s = VRP[:, 4]  # Service time
NUM_TW = VRP[:, 5]  # Number of time windows per location
r = VRP[:, 6::2]  # Ready times for each time window
e = VRP[:, 7::2]  # Due times for each time window
mx = VRP[:, 1]  # X-position of nodes
my = VRP[:, 2]  # Y-position of nodes

# Distance matrix
c = np.zeros((len(N), len(N)))
for i in N:
    for j in N:
        c[i][j] = int(math.sqrt((mx[j] - mx[i]) ** 2 + (my[j] - my[i]) ** 2))

# =========================================OPTIMIZATION MODEL========================================

m = Model('VRPmodel_MultiTW')

# Decision Variables
x = {}
for i in N:
    for j in N:
        for v in V:
            x[i, j, v] = m.addVar(vtype=GRB.BINARY, name=f"X_{i},{j},{v}")

z = {}
for i in N:
    for v in V:
        z[i, v] = m.addVar(vtype=GRB.BINARY, name=f"Z_{i},{v}")

t = {}
for i in N:
    for v in V:
        t[i, v] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"T_{i},{v}")

y = {}
for j in C:
    for t_index in range(NUM_TW[j]):
        for v in V:
            y[j, t_index, v] = m.addVar(vtype=GRB.BINARY, name=f"Y_{j},{t_index},{v}")

# Objective Function
obj = quicksum(c[i][j] * x[i, j, v] for i in N for j in N for v in V)
m.setObjective(obj, GRB.MINIMIZE)

# ========================================CONSTRAINTS===============================================

# Each location must be visited exactly once within one time window
for j in C:
    m.addConstr(quicksum(y[j, t, v] for t in range(NUM_TW[j]) for v in V) == 1)

# Link z and y variables
for j in C:
    for v in V:
        m.addConstr(z[j, v] == quicksum(y[j, t, v] for t in range(NUM_TW[j])))

# Depot constraints
m.addConstr(quicksum(z[0, v] for v in V) == len(V))
m.addConstr(quicksum(z[len(C) + 1, v] for v in V) == len(V))

# Route continuity
for h in C:
    for v in V:
        m.addConstr(quicksum(x[i, h, v] for i in N) == quicksum(x[h, j, v] for j in N))

# Capacity constraints
for v in V:
    m.addConstr(quicksum(d[j] * z[j, v] for j in C) <= Q)

# Time window constraints
for j in C:
    for t_index in range(NUM_TW[j]):
        for v in V:
            m.addConstr(y[j, t_index, v] * r[j, t_index] <= t[j, v])
            m.addConstr(t[j, v] <= y[j, t_index, v] * e[j, t_index])

# Travel time feasibility
for i in N:
    for j in N:
        if i != j:
            for v in V:
                m.addConstr(x[i, j, v] * (t[i, v] + c[i][j] + s[i] - t[j, v]) <= 0)

# Start from and return to the depot
for v in V:
    m.addConstr(quicksum(x[0, j, v] for j in N) == 1)
    m.addConstr(quicksum(x[i, len(C) + 1, v] for i in N) == 1)



m.update()
m.Params.timeLimit = 1800
m.optimize()

if m.status == GRB.Status.OPTIMAL:
    # Extract and display results
    arc_solution = m.getAttr('x', x)

    # Plot the routes
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(mx[1:-1], my[1:-1], c='blue')
    plt.scatter(mx[0], my[0], c='green', marker='s')  # Start depot
    plt.scatter(mx[-1], my[-1], c='red', marker='s')  # End depot

    for i in N:
        for j in N:
            for v in V:
                if arc_solution[i, j, v] > 0.99:
                    plt.plot([mx[i], mx[j]], [my[i], my[j]], 'r--')

    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.title("Vehicle Routes")
    plt.show()

    print(f"Objective Value: {m.objVal}")
