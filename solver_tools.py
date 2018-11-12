from math import acos, cos, sin, ceil, sqrt, atan2
import numpy as np
import itertools
import time


def getWorldActionsForCell(row, col, ugrids, vgrids, errors):

    numVectors = len(ugrids)

    uranges = [None for i in range(numVectors)]
    vranges = [None for i in range(numVectors)]

    numActions = 1
    for g in range(numVectors):
        uranges[g] = np.linspace(ugrids[g][row][col] - errors[g], ugrids[g][row][col] + errors[g], 10)
        vranges[g] = np.linspace(vgrids[g][row][col] - errors[g], vgrids[g][row][col] + errors[g], 10)
        numActions = numActions * len(uranges[g])

    uactionspace = list(itertools.product(*uranges))
    vactionspace = list(itertools.product(*vranges))

    world_actionspace = { "uactions"   : uactionspace,
                          "vactions"   : uactionspace,
                          "num"        : numActions,
                        }
    return world_actionspace



def getVectorSum(us, vs, weights):
    utotal = 0
    vtotal = 0

    numVectors = len(us)

    for i in range(numVectors):
        print (us[i], vs[i])
        utotal = utotal + weights[i] * us[i]
        vtotal = vtotal + weights[i] * vs[i]

    return (utotal, vtotal)


def getVectorDiff(us, vs, weights):
    utotal = 0
    vtotal = 0

    numVectors = len(us)

    for i in range(numVectors):
        utotal = utotal - weights[i] * us[i]
        vtotal = vtotal - weights[i] * vs[i]

    return utotal, vtotal


def magdir2uv(mag, dir_radians):
    u = mag * cos(dir_radians)
    v = mag * sin(dir_radians)
    return u, v

def uv2magdir(u, v):
    m = sqrt(u * u + v * v)
    d = atan2(v, u)
    return m, d

def getOutcome(move, us, vs, weights, traveler):

    # Get resultant vector of all weighted world sources
    uw, vw = getVectorSum(us, vs, weights)
    print("a", uw, vw)

    # Break traveler magnitude, direction into vector
    ut, vt = magdir2uv(traveler["speed_cps"], traveler["action2radians"][move])
    print("b", ut, vt)

    # Get vector difference (required applied force)
    ua, va = getVectorDiff([uw, ut], [vw, vt], [1, 1])
    print("c", ua, va)

    # Combine applied u, v to get mag, dir
    maga, dira = uv2magdir(ua, va)
    print("d", maga, dira)

    # Calculate work
    work = maga  # Proportional to magnitude the speed is 'per cell'
    print ("e", work)

    return work



def getGameForCell(row, col, traveler, ugrids, vgrids, errors, weights):

    world_actionspace = getWorldActionsForCell(row, col, ugrids, vgrids, errors)

    game = [[0 for wa in range(world_actionspace["num"])] for ta in traveler["actionspace"]]

    for row in range(len(game)):
        for col in range(len(game[0])):
            game[row][col] = getOutcome(traveler["actionspace"][row],
                    world_actionspace["uactions"][col], world_actionspace["vactions"][col],
                    weights, traveler)

    printGame(game)

    return game


def printGame(game):
    for g in game:
        print(g)


