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

    # Break traveler magnitude, direction into vector
    ut, vt = magdir2uv(traveler["speed_cps"], traveler["action2radians"][move])

    # Get vector difference (required applied force)
    ua, va = getVectorDiff([uw, ut], [vw, vt], [1, 1])

    # Combine applied u, v to get mag, dir
    maga, dira = uv2magdir(ua, va)

    # Calculate work
    work = maga  # Proportional to magnitude the speed is 'per cell'

    return work



def getGameForCell(row, col, traveler, ugrids, vgrids, errors, weights):

    world_actionspace = getWorldActionsForCell(row, col, ugrids, vgrids, errors)

    game = [[0 for wa in range(world_actionspace["num"])] for ta in traveler["actionspace"]]

    for row in range(len(game)):
        for col in range(len(game[0])):
            game[row][col] = getOutcome(traveler["actionspace"][row],
                    world_actionspace["uactions"][col], world_actionspace["vactions"][col],
                    weights, traveler)

    game = np.array(game)

    return game


def solve_williams(payoff_matrix, iterations=200):
    '''
    Source: http://code.activestate.com/recipes/496825-game-theory-payoff-matrix-solver/

    Approximate the strategy oddments for 2 person zero-sum games of perfect information.

    Applies the iterative solution method described by J.D. Williams in his classic
    book, The Complete Strategyst, ISBN 0-486-25101-2.   See chapter 5, page 180 for details.
    '''

    from operator import add, neg
    'Return the oddments (mixed strategy ratios) for a given payoff matrix'
    transpose = zip(*payoff_matrix)
    numrows = len(payoff_matrix)
    numcols = len(transpose)
    row_cum_payoff = [0] * numrows
    col_cum_payoff = [0] * numcols
    colpos = range(numcols)
    rowpos = map(neg, xrange(numrows))
    colcnt = [0] * numcols
    rowcnt = [0] * numrows
    active = 0
    for i in xrange(iterations):
        rowcnt[active] += 1
        col_cum_payoff = map(add, payoff_matrix[active], col_cum_payoff)
        active = min(zip(col_cum_payoff, colpos))[1]
        colcnt[active] += 1
        row_cum_payoff = map(add, transpose[active], row_cum_payoff)
        active = -max(zip(row_cum_payoff, rowpos))[1]
    value_of_game = (max(row_cum_payoff) + min(col_cum_payoff)) / 2.0 / iterations
    return np.array(rowcnt).astype(float) / iterations, np.array(colcnt).astype(float) / iterations

def solve_gambit(game):
    import gambit

    m = game.shape[0]
    n = game.shape[1]

    g = gambit.Game.new_table([m, n])
    g.title = "solve cell"
    g.players[0].label = "traveler"
    g.players[1].label = "fujin"

    # Populate game
    for row in range(m):
        for col in range(n):
            g[row, col][0] = int(game[row][col] * 1000)
            g[row, col][1] = (-1) * int(game[row][col] * 1000)

    solver = gambit.nash.ExternalEnumMixedSolver()
    solution = solver.solve(g)
    solution = np.asarray(solution[0])

    y = solution[0:n-1]
    z = solution[n-1:]

    return y, z

def solveGame(game, method = 0):

    '''
       methods
       ------------
       0 : williams
       1 : gambit
    '''

    if method == 0:
        y, z  = solve_williams(game)
    if method == 1:
        y, z  = solve_gambit(game)

    # Convert to pure policy
    i = np.argmax(y)
    j = np.argmax(z)
    v = game[i][j]

    return (y, z, i, j, v)


def printGame(game):
    for row in range(len(game)):
        for col in range(len(game[0])):
            print("{0:.2f}".format(game[row][col])),
        print ("")

def printSolution(solution, name = ""):
    print("Nash Solution " + name)
    print("Player 1:" )
    print("    Pure sucurity policy: {}".format(solution[2]))
    print("    Security policy: ")
    print("    "),
    for num in solution[0]:
        print("{} ".format(num)),
    print("")
    print("Player 1:" )
    print("    Pure sucurity policy: {}".format(solution[3]))
    print("    Security policy: ")
    print("    "),
    for num in solution[1]:
        print("{} ".format(num)),
    print("")
    print("Security level: {}".format(solution[4]))



def getNashGrid(traveler, occgrid, ugrids, vgrids, errors, weights):

    nashgrid = np.zeros(occgrid.shape) - np.inf
    m, n = nashgrid.shape


    for row in range(m):
        for col in range(n):
            if (occgrid[row][col] == 0):
                g = getGameForCell(row, col, traveler, ugrids, vgrids, errors, weights)
                solution = solveGame(g, 0)
                nashgrid[row][col] = solution[4]
    return nashgrid




def getCost2go(traveler, nashgrid, occgrid, errors, weights, iterations = 100):

    def haltCost(row, col, target_row, target_col):

        # INF cost if not halting at target
        if  row != target_row or \
            col != target_col:
                return np.inf
        else: # No cost to halt at target
            return 0


    def floodAssignCost(row, col, nashgrid, cost2go, actiongrid, traveler):
        stack = set(((row, col),))

        visitedgrid = np.zeros(nashgrid.shape)

        while stack:
            row, col = stack.pop()

            # If not obstacle..
            if nashgrid[row][col] > - np.inf:

                # Assign
                mincost = np.inf
                action = None
                if row > 0:
                    v = cost2go[row - 1][col]
                    if v < mincost:
                        mincost = v
                        action  = "U"
                if row < len(nashgrid) - 1:
                    v = cost2go[row + 1][col]
                    if v < mincost:
                        mincost = v
                        action  = "D"
                if col > 0:
                    v = cost2go[row][col - 1]
                    if v < mincost:
                        mincost = v
                        action  = "L"
                if col < len(nashgrid[0]) - 1:
                    v = cost2go[row][col + 1]
                    if v < mincost:
                        mincost = v
                        action  = "R"
                v = haltCost(row, col, traveler["target"][0], traveler["target"][1])
                if v < mincost:
                    mincost = v
                    action  = "H"

                if row == traveler["target"][0] and \
                   col == traveler["target"][1]:
                    cost2go[row][col] = 0
                    action = "H"
                else:
                    #cost2go[row][col] = mincost + 1                        # <- uncomment to ignore work
                    cost2go[row][col] = mincost + (10) * nashgrid[row][col] # <- uncomment to include work

                actiongrid[row][col] = action

                visitedgrid[row][col] = 1

                # recursion
                if row > 0:
                    if visitedgrid[row - 1][col] == 0:
                       stack.add((row - 1, col))
                if row < len(nashgrid) - 1:
                    if visitedgrid[row + 1][col] == 0:
                        stack.add((row + 1, col))
                if col > 0:
                    if visitedgrid[row][col - 1] == 0:
                        stack.add((row, col - 1))
                if col < len(nashgrid[0]) - 1:
                    if visitedgrid[row][col + 1] == 0:
                        stack.add((row, col + 1))


    m, n = nashgrid.shape

    cost2go = np.zeros(occgrid.shape) + np.inf
    actiongrid = [["x" for col in range(n)] for row in range(m)]

    for i in range(iterations):
        floodAssignCost(traveler["target"][0], traveler["target"][1], nashgrid, cost2go, actiongrid, traveler)

    print(cost2go)

    return cost2go, actiongrid



def writeActiongrid(actiongrid, actionfile):

    numrows = len(actiongrid)
    numcols = len(actiongrid[0])

    action2symbol = { "U" : "^", "D" : "v", "L" : "<", "R" : ">",
                      "H" : "#", "x" : "-", }

    af = open(actionfile, 'w')
    for row in range(numrows):
        for col in range(numcols):
            af.write("%s" % action2symbol[actiongrid[row][col]])
        af.write("\n")

    af.close()




