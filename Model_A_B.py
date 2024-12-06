#==================================================================================================
from gurobipy import *
import numpy as np
import math
import copy
import pandas as pd
import matplotlib.pyplot as plt

#from matplotlib import rc
#rc('text', usetex=True)
#import os
#os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'

#============================================MODEL DATA============================================

with open("data_small.txt", "r") as f:  # Open Li & Lim PDPTW instance definitions
    data = f.readlines()  # Extract instance definitions

VRP = []  # Create array for data related to nodes
i = 0  # Varible to keep track of lines in data file
for line in data:
    i = i + 1
    words = line.split()
    words = [int(i) for i in words]  # Covert data from string to integer
    VRP.append(words)  # Store node data
VRP.append(VRP[-0][:])
VRP[len(VRP)-1][0] = len(VRP)-1
VRP = np.array(VRP)
'''
Column 0 -> location ID
Column 1 -> x
Column 2 -> y
Column 3 -> demand
Column 4 -> service time
Column 5 -> ready time
Column 6 -> due (end) time
'''

#===============================PARAMETERS===============================================

# N = total number of nodes, including the depot twice
# C = total number of locations to serve, excluding the depot twice
# N = C + 2
N = VRP[:, 0]       # Nodes
C = VRP[1:-1, 0]    # Customers/Locations
V = [0, 1]          # Vehicles
# q = [130, 130]    # unique vehicle capacities
Q = 130             # all vehicles same capacity
d = VRP[:, 3]
s = VRP[:, 4]
r = VRP[:, 5]
e = VRP[:, 6]
mx = VRP[:, 1]  # X-position of nodes
my = VRP[:, 2]  # Y-position of nodes
c = np.zeros((len(N), len(N)))  # Create array for distance between nodes
for i in N:
    for j in N:
        c[i][j] = int(math.sqrt((mx[j] - mx[i]) ** 2 + (my[j] - my[i]) ** 2))

#================================OPTIMIZATION MODEL========================================

## Create optimization model
m = Model('VRPmodel')

## Create Decision Variables
# if x(i, j, v) = 1, loc j is visited from loc i by vehicle v
x = {}
for i in N:
    for j in N:
        for v in V:
            x[i, j, v] = m.addVar(vtype=GRB.BINARY, lb=0, name="X_%s,%s,%s" % (i, j, v))

# if z(i, v) = 1, loc i is visited by vehicle v
z = {}
for i in N:
    for v in V:
        z[i, v] = m.addVar(vtype=GRB.BINARY, lb=0, name="Z_%s,%s" % (i, v))

# point of time of vehicle v at loc i
t = {}
for i in N:
    for v in V:
        t[i, v] = m.addVar(vtype=GRB.CONTINUOUS, lb=0, name="T_%s,%s" % (i, v))

#============================OBJECTIVE FUNCTION=====================================================

obj = quicksum(c[i][j] * x[i, j, v] for i in N for j in N for v in V)
m.setObjective(obj, GRB.MINIMIZE)

#============================CONSTRAINTS============================================================

# All locations should be visited only once
for i in C:
    m.addConstr(quicksum(z[i, v] for v in V) == 1, name='Visit_%s' % i)

# All vehicles should have the depot in their route at the start and also end at the depot (C + 1)
m.addConstr(quicksum(z[0, v] for v in V) == len(V), name='VisitDepot_%s' % 0)
m.addConstr(quicksum(z[len(C)+1, v] for v in V) == len(V), name='VisitDepot2_%s' % 0)

# All vehicles that go into a loc, also have to leave the loc (in--->loc--->out)
for h in C:
    for v in V:
        m.addConstr(quicksum(x[i, h, v] for i in N) == quicksum(x[h, j, v] for j in N), name='ArcIn_%s,%s' % (i, v))

# Location only visited from one node (also configures z to x)
for j in N:
    for v in V:
        m.addConstr(quicksum(x[i, j, v] for i in N) == z[j, v], name='ArcOut_%s,%s' % (i, v))

# Capacity constraints
for v in V:
    # m.addConstr(quicksum(d[i] * z[i, v] for i in N) <= q[v], name='Capacity_%s' % v)
    m.addConstr(quicksum(d[i] * z[i, v] for i in C) <= Q, name='Capacity_%s' % v)

# Time window constraints
for i in N:
    for v in V:
        m.addConstr(t[i, v] >= r[i], name='ready_%s,%s' % (i, v))
        m.addConstr(t[i, v] <= e[i], name='end_%s,%s' % (i, v))

# If if x(i, j, v) = 1 (loc j is visited from loc i by vehicle v) then t[j, v] is t[i, v] + costs and service time
for i in N:
    for j in N:
        for v in V:
            m.addConstr(x[i, j, v] * (t[i, v] + c[i][j] + s[i] - t[j, v]) <= 0, name='InTime_%s,%s,%s' % (i, j, v))

m.update()
m.write('VRPModel.lp')
m.Params.timeLimit = 1800

m.optimize()
# m.computeIIS()
# m.write('iismodel.ilp')
if m.status == GRB.Status.OPTIMAL:

    m.write('VRPModel.sol')

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
    # plt.plot(mx[0], my[0], c='g', marker='s')
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
    for v in V:
        Totaldistance = sum(c[i][j] * x[i, j, v].X for i in N for j in N)
        print('Total distance traveled for vehicle ' + str(v) + ': ', Totaldistance)