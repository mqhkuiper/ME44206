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
VRP.append(VRP[-0][:])
VRP[len(VRP)-1][0] = len(VRP)-1
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
for j in N:
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
for j in N:
    for v in V:
        m.addConstr(z[j, v] == quicksum(y[j, t, v] for t in range(NUM_TW[j])))

# Depot constraints
m.addConstr(quicksum(z[0, v] for v in V) == len(V))
m.addConstr(quicksum(z[len(C) + 1, v] for v in V) == len(V))

for j in N:
    for v in V:
        m.addConstr(quicksum(x[i, j, v] for i in N) == z[j, v], name='ArcOut_%s,%s' % (i, v))

# Route continuity
for h in C:
    for v in V:
        m.addConstr(quicksum(x[i, h, v] for i in N) == quicksum(x[h, j, v] for j in N))

# Capacity constraints
for v in V:
    m.addConstr(quicksum(d[j] * z[j, v] for j in C) <= Q)

# Time window constraints
for j in N:
    for v in V:
        for t_index in range(NUM_TW[j]):
            m.addConstr(y[j, t_index, v] * t[j, v] >= y[j, t_index, v] * r[j][t_index])
            m.addConstr(y[j, t_index, v] * t[j, v] <= y[j, t_index, v] * e[j][t_index])

# Travel time feasibility
for i in N:
    for j in N:
        for v in V:
            m.addConstr(x[i, j, v] * (t[i, v] + c[i][j] + s[i] - t[j, v]) <= 0)

m.update()
m.Params.timeLimit = 1800
m.optimize()

if m.status == GRB.Status.OPTIMAL:

    m.write('VRPModelTW.sol')

    # Plot the routes that are decided to be traversed
    arc_solution = m.getAttr('x', x)

    fig = plt.figure(figsize=(15, 15))
    plt.xlabel('x-coordinate')
    plt.ylabel('y-coordinate')
    plt.scatter(mx[0:len(N)], my[0:len(N)])
    for i in range(0, len(N)):
        if i == 0 or i == len(N)-1:
            plt.annotate('Depot', (mx[i], my[i]), fontsize=20)
        else:
            plt.annotate(str(i), (mx[i], my[i]), fontsize=20)
    #
    path_colors = ['r--', 'b--', 'g--', 'm--', 'c--']
    for i in range(len(N)):
        for j in range(len(N)):
            for v in range(len(V)):
                if arc_solution[i, j, v] > 0.99:
                    plt.plot([mx[i], mx[j]], [my[i], my[j]], path_colors[v], label='Vehicle ' + str(v + 1))
                    handles, labels = plt.gca().get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    plt.legend(by_label.values(), by_label.keys())
    plt.show()
    #YOU CAN SAVE YOUR PLOTS SOMEWHERE IF YOU LIKE
    #plt.savefig('Plots/TSP.png',bbox_inches='tight')

    for v in V:
        route = m.getAttr('x', t)
        route_picked = m.getAttr('x', z)
        route_array = []
        for i in N:
            if route[i, v] > 0.99 and route_picked[i, v] > 0.99:
                route_array.append([i, t[i, v].X])
        route_sorted = sorted(route_array, key=lambda ele: ele[1])
        load = 0
        for i in route_sorted:
            loc = i[0]
            i.append(load)
            load += d[loc]
            if loc == len(C) + 1:
                i[0] = 'Depot'
        conc_str = 'Loc-Depot>' + '>'.join('Loc' + str(a) + ' visited at T' + str(b) + ' with load Q' + str(c) for a, b, c in route_sorted)
        print("Node route of vehicle ", v, " is ", conc_str)
    print('Obj: %g' % m.objVal)
    Totaldistance = sum(c[i][j] * x[i, j, v].X for i in N for j in N for v in V)
    print('Total distance traveled: ', Totaldistance)