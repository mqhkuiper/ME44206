#================================================================================
#Bilge Atasoy
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
N = VRP[:, 0]       # Nodes
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

#========================================OPTIMIZATION MODEL========================================


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

## Objective
obj = quicksum(c[i][j] * x[i, j, v] for i in N for j in N for v in V)
m.setObjective(obj, GRB.MINIMIZE)

# Constraints    
# All locations should be visited only once
for i in N[1: len(N)]:
    m.addConstr(quicksum(z[i, v] for v in V) == 1, name='Visit_%s' % i)

# All vehicles should have the depot in their route
m.addConstr(quicksum(z[0, v] for v in V) == len(V), name='VisitDepot_%s' % 0)

# Capacity constraints
for v in V:
    # m.addConstr(quicksum(d[i] * z[i, v] for i in N) <= q[v], name='Capacity_%s' % v)
    m.addConstr(quicksum(d[i] * z[i, v] for i in N) <= Q, name='Capacity_%s' % v)

# All vehicles that go into a loc, also have to leave the loc (in--->loc--->out)
for j in N:
    for v in V:
        m.addConstr(quicksum(x[i, j, v] for i in N) == z[j, v], name='ArcIn_%s,%s' % (j, v))
for i in N:
    for v in V:
        m.addConstr(quicksum(x[i, j, v] for j in N) == z[i, v], name='ArcOut_%s,%s' % (i, v))

# Time window constraints
for i in N:
    for v in V:
        m.addConstr(t[i, v] >= r[i], name='ready_%s,%s' % (i, v))
        m.addConstr(t[i, v] <= e[i], name='end_%s,%s' % (i, v))

# If if x(i, j, v) = 1 (loc j is visited from loc i by vehicle v) then t[j, v] is t[i, v] + costs and service time
for i in N:
    for j in N[1: len(N)]:
        for v in V:
            m.addConstr(x[i, j, v] * (t[i, v] + c[i][j] + s[i] - t[j, v]) <= 0, name='InTime_%s,%s,%s' % (i, j, v))
            # m.addConstr(t[i, v] + c[i][j] + s[i] - 100000*(1 - x[i, j, v]) <= t[j, v], name='%s,%s,%s' % (i, j, v))

# Ensure vehicle comes back before depot closing time
for i in N:
    for v in V:
        m.addConstr(x[i, 0, v] * (t[i, v] + c[i][0] + s[i]) <= e[0], name='VisitDepot_%s' % 0)

m.update()
m.write('VRPModel.lp')
m.Params.timeLimit = 36000

m.optimize()
# m.computeIIS()
# m.write('iismodel.ilp')
if m.status == GRB.Status.OPTIMAL:

    m.write('VRPModel.sol')

    # Plot the routes that are decided to be traversed
    arc_solution = m.getAttr('x', x)
    #
    fig = plt.figure(figsize=(15, 15))
    plt.xlabel('x-coordinate')
    plt.ylabel('y-coordinate')
    plt.scatter(mx[1:len(N) - 1], my[1:len(N) - 1])
    for i in range(1, len(N) - 1):
        plt.annotate(str(i), (mx[i], my[i]))
    plt.plot(mx[0], my[0], c='g', marker='s')
    #
    for i in range(len(N)):
        for j in range(len(N)):
            for v in range(len(V)):
                if arc_solution[i, j, v] > 0.99:
                    plt.plot([mx[i], mx[j]], [my[i], my[j]], 'r--')
    plt.show()
    #YOU CAN SAVE YOUR PLOTS SOMEWHERE IF YOU LIKE
    #plt.savefig('Plots/TSP.png',bbox_inches='tight')


    # for i in V:
    #     print("ORDER OF NODE ", i, " IS ", u[i].X)

    print('Obj: %g' % m.objVal)

    Totaldistance = sum(c[i][j] * x[i, j, v].X for i in N for j in N for v in V)

    print('Total distance traveled: ', Totaldistance)
