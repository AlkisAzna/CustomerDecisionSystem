#!/usr/bin/env python
# coding: utf-8

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp

try:

    # Create a new model
    m = gp.Model("CustomerDecision")

    # Create variables
    s = np.empty(42, dtype=object)
    for index, element in enumerate(s):
        s[index] = m.addVar(vtype=GRB.BINARY, name="s"+str(index+1))

    w11 = m.addVar(vtype=GRB.BINARY, name="w11")
    w12 = m.addVar(vtype=GRB.BINARY, name="w12")
    w13 = m.addVar(vtype=GRB.BINARY, name="w13")
    w21= m.addVar(vtype=GRB.BINARY, name="w21")
    w22 = m.addVar(vtype=GRB.BINARY, name="w22")
    w23 = m.addVar(vtype=GRB.BINARY, name="w23")
    w31 = m.addVar(vtype=GRB.BINARY, name="w31")
    w32 = m.addVar(vtype=GRB.BINARY, name="w32")
    w41 = m.addVar(vtype=GRB.BINARY, name="w41")
    w42 = m.addVar(vtype=GRB.BINARY, name="w42")
    w51 = m.addVar(vtype=GRB.BINARY, name="w51")
    w52 = m.addVar(vtype=GRB.BINARY, name="w52")
    w53 = m.addVar(vtype=GRB.BINARY, name="w53")
    w61 = m.addVar(vtype=GRB.BINARY, name="w61")
    w62 = m.addVar(vtype=GRB.BINARY, name="w62")
    w63 = m.addVar(vtype=GRB.BINARY, name="w63")
    w64 = m.addVar(vtype=GRB.BINARY, name="w64")
    
    # Set objective
    m.setObjective(sum(s), GRB.MINIMIZE)

    # Add constraints
    # DM1
    m.addConstr(w11 + w12 + 1.0 * w13 + w21 + w22 + 1.0 * w23 + w31 + 1.0 * w32 + w61 + w62 + 1.0 * w63 - 
                (1.0 * w21 + 1.0 * w41 + w51 + 0.5 * w52 + 0.0) - s[0] + s[1] + s[2] - s[3] >= 0.05, "c0")
    m.addConstr(1.0 * w21 + 1.0 * w41 + w51 + 0.5 * w52 + 0.0 - 
                (0.5 * w11 + 0.4 * w21 + w31 + 0.33 * w32 + 0.5 * w41 + w51 + w52 + 0.1 * w53 + 1.0 * w61) - s[2] + s[3] + s[4] - s[5] >= 0.05, "c1")
    m.addConstr(0.5 * w11 + 0.4 * w21 + w31 + 0.33 * w32 + 0.5 * w41 + w51 + w52 + 0.1 * w53 + 1.0 * w61 - 
                (1.0 * w11 + w31 + 1.0 * w32 + w41 + 1.0 * w42 + w51 + w52 + 1.0 * w53 + 0.0) - s[4] + s[5] + s[6] - s[7] >= 0.05, "c2")
    m.addConstr(1.0 * w11 + w31 + 1.0 * w32 + w41 + 1.0 * w42 + w51 + w52 + 1.0 * w53 + 0.0 - 
                (w11 + 1.0 * w12 + w21 + w22 + 1.0 * w23 + w31 + 0.33 * w32 + 1.0 * w41 + w51 + 0.9 * w52 + w61 + w62 + w63 + 1.0 * w64) - s[6] + s[7] + s[8] - s[9] >= 0.05, "c3")
    m.addConstr(w11 + 1.0 * w12 + w21 + w22 + 1.0 * w23 + w31 + 0.33 * w32 + 1.0 * w41 + w51 + 0.9 * w52 + w61 + w62 + w63 + 1.0 * w64 - 
                (w11 + w12 + 0.5 * w13 + w21 + w22 + 0.4 * w23 + 1.0 * w31 + w51 + 0.5 * w52 + w61 + w62 + 1.0 * w63) - s[8] + s[9] + s[10] - s[11] >= 0.05, "c4")
    m.addConstr(w11 + w12 + 0.5 * w13 + w21 + w22 + 0.4 * w23 + 1.0 * w31 + w51 + 0.5 * w52 + w61 + w62 + 1.0 * w63 - 
                (0.5 * w11 + w21 + 1.0 * w22 + w31 + 0.67 * w32 + 0.5 * w41 + w51 + w52 + 0.2 * w53 + w61 + 1.0 * w62) - s[10] + s[11] + s[12] - s[13] == 0.0, "c5")
    
    # DM2
    m.addConstr(1.0 * w11 + w31 + 1.0 * w32 + w41 + 1.0 * w42 + w51 + w52 + 1.0 * w53 + 1.0 * w61 - 
                (0.5 * w11 + 0.4 * w21 + w31 + 0.33 * w32 + 0.5 * w41 + w51 + w52 + 0.1 * w53 + w61 + w62 + w63 + 1.0 * w64) - s[14] + s[15] + s[16] - s[17] >= 0.05, "c6")
    m.addConstr(0.5 * w11 + 0.4 * w21 + w31 + 0.33 * w32 + 0.5 * w41 + w51 + w52 + 0.1 * w53 + w61 + w62 + w63 + 1.0 * w64 - 
                (w11 + 1.0 * w12 + w21 + w22 + 1.0 * w23 + w31 + 0.33 * w32 + 1.0 * w41 + w51 + 0.9 * w52 + w61 + 1.0 * w62) - s[16] + s[17] + s[18] - s[19] == 0.0, "c7")
    m.addConstr(w11 + 1.0 * w12 + w21 + w22 + 1.0 * w23 + w31 + 0.33 * w32 + 1.0 * w41 + w51 + 0.9 * w52 + w61 + 1.0 * w62 - 
                (0.5 * w11 + w21 + 1.0 * w22 + w31 + 0.67 * w32 + 0.5 * w41 + w51 + w52 + 0.2 * w53 + w61 + 1.0 * w62) - s[18] + s[19] + s[20] - s[21] >= 0.05, "c8")
    m.addConstr(0.5 * w11 + w21 + 1.0 * w22 + w31 + 0.67 * w32 + 0.5 * w41 + w51 + w52 + 0.2 * w53 + w61 + 1.0 * w62 - 
                (w11 + w12 + 0.5 * w13 + w21 + w22 + 0.4 * w23 + 1.0 * w31 + w51 + 0.5 * w52 + 0.0) - s[20] + s[21] + s[22] - s[23] >= 0.05, "c9")
    m.addConstr(w11 + w12 + 0.5 * w13 + w21 + w22 + 0.4 * w23 + 1.0 * w31 + w51 + 0.5 * w52 + 0.0 - 
                (1.0 * w21 + 1.0 * w41 + w51 + 0.5 * w52 + w61 + w62 + 1.0 * w63) - s[22] + s[23] + s[24] - s[25] == 0.0, "c10")
    m.addConstr(1.0 * w21 + 1.0 * w41 + w51 + 0.5 * w52 + w61 + w62 + 1.0 * w63 - 
                (w11 + w12 + 1.0 * w13 + w21 + w22 + 1.0 * w23 + w31 + 1.0 * w32 + w61 + w62 + 1.0 * w63) - s[24] + s[25] + s[26] - s[27] >= 0.05, "c11") 
    
    # DM3
    m.addConstr(0.5 * w11 + 0.4 * w21 + w31 + 0.33 * w32 + 0.5 * w41 + w51 + w52 + 0.1 * w53 + w61 + w62 + w63 + 1.0 * w64 -
                (1.0 * w11 + w31 + 1.0 * w32 + w41 + 1.0 * w42 + w51 + w52 + 1.0 * w53 + 0.0) - s[28] + s[29] + s[30] - s[31] >= 0.05, "c12")
    m.addConstr(1.0 * w11 + w31 + 1.0 * w32 + w41 + 1.0 * w42 + w51 + w52 + 1.0 * w53 + 0.0 - 
                (w11 + w12 + 1.0 * w13 + w21 + w22 + 1.0 * w23 + w31 + 1.0 * w32 + w61 + w62 + 1.0 * w63) - s[30] + s[31] + s[32] - s[33] >= 0.05, "c13")
    m.addConstr(w11 + w12 + 1.0 * w13 + w21 + w22 + 1.0 * w23 + w31 + 1.0 * w32 + w61 + w62 + 1.0 * w63 - 
                (1.0 * w21 + 1.0 * w41 + w51 + 0.5 * w52 + 1.0 * w61) - s[32] + s[33] + s[34] - s[35] == 0.0, "c14")
    m.addConstr(1.0 * w21 + 1.0 * w41 + w51 + 0.5 * w52 + 1.0 * w61 - 
                (w11 + 1.0 * w12 + w21 + w22 + 1.0 * w23 + w31 + 0.33 * w32 + 1.0 * w41 + w51 + 0.9 * w52 + w61 + w62 + 1.0 * w63) - s[34] + s[35] + s[36] - s[37] >= 0.05, "c15")
    m.addConstr(w11 + 1.0 * w12 + w21 + w22 + 1.0 * w23 + w31 + 0.33 * w32 + 1.0 * w41 + w51 + 0.9 * w52 + w61 + w62 + 1.0 * w63 - 
                (0.5 * w11 + w21 + 1.0 * w22 + w31 + 0.67 * w32 + 0.5 * w41 + w51 + w52 + 0.2 * w53 + w61 + 1.0 * w62) - s[36] + s[37] + s[38] - s[39] >= 0.05, "c16")
    m.addConstr(0.5 * w11 + w21 + 1.0 * w22 + w31 + 0.67 * w32 + 0.5 * w41 + w51 + w52 + 0.2 * w53 + w61 + 1.0 * w62 - 
                (w11 + w12 + 0.5 * w13 + w21 + w22 + 0.4 * w23 + 1.0 * w31 + w51 + 0.5 * w52 + 1.0 * w61) - s[38] + s[39] + s[40] - s[41] >= 0.05, "c17") 

    
    # S and W constraints
    for index, element in enumerate(s):
        m.addConstr(s+(index+1) >= 0.0, "c"+str(index+18))
    
    m.addConstr(w11 >= 0.0, "c60")
    m.addConstr(w12 >= 0.0, "c61")
    m.addConstr(w13 >= 0.0, "c62")
    m.addConstr(w21 >= 0.0, "c63")
    m.addConstr(w22 >= 0.0, "c64")
    m.addConstr(w23 >= 0.0, "c65")
    m.addConstr(w31 >= 0.0, "c66")
    m.addConstr(w32 >= 0.0, "c67")
    m.addConstr(w41 >= 0.0, "c68")
    m.addConstr(w42 >= 0.0, "c69")
    m.addConstr(w51 >= 0.0, "c70")
    m.addConstr(w52 >= 0.0, "c71")
    m.addConstr(w53 >= 0.0, "c72")
    m.addConstr(w61 >= 0.0, "c73")
    m.addConstr(w62 >= 0.0, "c74")
    m.addConstr(w63 >= 0.0, "c75")
    m.addConstr(w64 >= 0.0, "c76")
    
    # Optimize model
    m.optimize()
    
except gp.GurobiError as e:
    print('Error code ' + str(e.errno) + ": " + str(e))

except AttributeError:
    print('Encountered an attribute error')




