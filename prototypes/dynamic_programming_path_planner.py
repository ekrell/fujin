# Implementation of:
#     K. S. Kwok and B. J. Driessen,
#     "Path planning for complex terrain navigation via
#      dynamic programming"
#     Proceedings of the 1999 American Control Conference
#     1999, pp. 2941-2944 vol.4.
#     doi: 10.1109/ACC.1999.786612
#
# Author: Evan Krell

import numpy as np

# Matrix M: Occupancy grid
# M[i][j] = 0 if obstacle, M[i][j] = 1 if free
# Size (15, 15)
M = np.array([
      [ 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0 ],
      [ 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1 ],
      [ 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1 ],
      [ 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ],
      [ 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1 ],
      [ 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1 ],
      [ 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1 ],
      [ 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1 ],
      [ 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1 ],
      [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1 ],
      [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1 ],
      [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1 ],
      [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1 ],
      [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ],
      [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ],
    ])

# Maximum indices
i_max = len(M) - 1
j_max = len(M[0]) - 1

# Tensor C: Cost2go
# C[k][i][j] is cost2go for kth time step back from final time to (i, j) location
# Initialized to size (1, 15, 15) since it must expand dynamically.
C = [
    [ [ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 ],
      [ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 ],
      [ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 ],
      [ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 ],
      [ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 ],
      [ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 ],
      [ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 ],
      [ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 ],
      [ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 ],
      [ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 ],
      [ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 ],
      [ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 ],
      [ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 ],
      [ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 ],
      [ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 ],
    ],
    ]

# Target location
i_end = 13
j_end = 13
# Set only target as valid for the k = 1 time step
C[0][i_end][j_end] = 0

# Vector A of possible actions:
# A[i] is one of the 8 directions:
# 0 -> UP, 1 -> DOWN, 2 -> LEFT, 3 -> RIGHT
# 4 -> UP-LEFT, 5 -> UP-RIGHT, 6 -> DOWN-LEFT, 7 -> DOWN-RIGHT
A = range(9)

# Function distance: destance between locations
def distance(i1, j1, i2, j2):
    return pow((pow(i1 - i2, 2) + pow(j1 - j2, 2)), 0.5)

# Function L1: destination row if using action m
def L1(i, j, m):
    if   m == 0:
        return i - 1
    elif m == 1:
        return i + 1
    elif m == 2:
        return i
    elif m == 3:
        return i
    elif m == 4:
        return i - 1
    elif m == 5:
        return i - 1
    elif m == 6:
        return i + 1
    elif m == 7:
        return i + 1

# Function L1: destination row if using action m
def L2(i, j, m):
    if   m == 0:
        return j
    elif m == 1:
        return j
    elif m == 2:
        return j - 1
    elif m == 3:
        return j + 1
    elif m == 4:
        return j - 1
    elif m == 5:
        return j + 1
    elif m == 6:
        return j - 1
    elif m == 7:
        return j + 1

# Function F: Cost to perform action
def F(i, j, m, i_max, j_max):
    i_new = L1(i, j, m)
    j_new = L2(i, j, m)
    if i_new < 0 or i_new > i_max or j_new < 0 or j_new > j_max or M[i_new][j_new] == 0:
        return -1
    else:
        return distance(i, j, i_new, j_new)

# Set S: allowed moves
# S[k][i][j] is the feasible moves from location (i, j) at backward time step k
S = []

# Set I: optimial coordinates
# I[k][i][j] is the optimimal i coordinate from location (i, j) at backward time step k
I = []
# Set J: optimial coordinates
# J[k][i][j] is the optimimal j coordinate from location (i, j) at backward time step k
J = []


limit = 10
k = 0
while k < limit:

    # Update S for backward time step k
    S_k = [[[] for j in range(j_max)] for i in range(i_max)]
    for i in range(i_max):
        for j in range(j_max):
            for m in A:
                i_new = L1(i, j, m)
                j_new = L2(i, j, m)
                if i_new == None or j_new == None:
                    continue
                if F(i, j, m, i_max, j_max) != -1:
                    if C[k][i_new][j_new] != -1:
                        S_k[i][j].append(m)
    S.append(S_k)

    # Update C for backward time step k+1
    C_kp1 = [[np.inf for j in range(j_max + 1)] for i in range(i_max + 1)]
    I_kp1 = [[-1 for j in range(j_max + 1)] for i in range(i_max + 1)]
    J_kp1 = [[-1 for j in range(j_max + 1)] for i in range(i_max + 1)]

    m_select = None
    for i in range(i_max):
        for j in range(j_max):
            if len(S[k][i][j]) == 0:
                C_kp1[i][j] = -1
            else:
                for m in A:
                    i_new = L1(i, j, m)
                    j_new = L2(i, j, m)
                    if i_new == None or j_new == None:
                        continue
                    c_new = F(i, j, m, i_max, j_max) + C[k][i_new][j_new]
                    if c_new < C_kp1[i][j]:
                        C_kp1[i][j] = c_new
                        m_select = m
                        I_kp1[i][j] = i_new
                        J_kp1[i][j] = j_new
    C.append(C_kp1)
    I.append(I_kp1)
    J.append(J_kp1)

    k = k + 1


print(np.array(I[9]))
