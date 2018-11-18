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

def getOutcome(move, us, vs, weights, traveler, cost2go = None, row = None, col = None):

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
    # Dynamic programming: include cost2go
    if cost2go is not None and row is not None and col is not None:

        if move == "U":
            if row > 0:
                work = work + cost2go[row - 1][col]
            else:
                work = work + 100000000000
        if move == "D":
            if row < len(cost2go) - 1:
                work = work + cost2go[row + 1][col]
            else:
                work = work + 100000000000
        if move == "L":
            if col > 0:
                work = work + cost2go[row][col - 1]
            else:
                work = work + 100000000000
        if move == "R":
            if col < len(cost2go[0]) - 1:
                work = work + cost2go[row][col + 1]
            else:
                work = work + 100000000000
        if move == "H":
            work = 100000000000

    return work

def getGameForCell(row, col, traveler, ugrids, vgrids, errors, weights, cost2go = None):

    world_actionspace = getWorldActionsForCell(row, col, ugrids, vgrids, errors)

    game = [[0 for wa in range(world_actionspace["num"])] for ta in traveler["actionspace"]]

    for r in range(len(game)):
        for c in range(len(game[0])):
            game[r][c] = getOutcome(traveler["actionspace"][r],
                    world_actionspace["uactions"][c], world_actionspace["vactions"][c],
                    weights, traveler, cost2go, row, col)

    game = np.array(game)

    return game


def solve_williams(payoff_matrix, iterations=100):
    '''
    Source: http://code.activestate.com/recipes/496825-game-theory-payoff-matrix-solver/

    Approximate the strategy oddments for 2 person zero-sum games of perfect information.

    Applies the iterative solution method described by J.D. Williams in his classic
    book, The Complete Strategyst, ISBN 0-486-25101-2.   See chapter 5, page 180 for details.
    '''

    payoff_matrix = (-1) * payoff_matrix

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

    #game = (-1) * np.array([ [10, 0], [0, 0] ])
    game = game

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
    y = solution[0:m]
    z = solution[m:m+n]
    return y, z

def solve_nashpy(game):
    import nashpy as nash

    rps = nash.Game(game)
    eqs = list(rps.support_enumeration())

    y = eqs[0]
    z = eqs[1]

    return y, z

def solveGame(game, method = 0):

    '''
       methods
       ------------
       0 : williams
       1 : gambit
       2 : nashpy
    '''

    if   method == 0:
        y, z = solve_williams(game)
    elif method == 1:
        y, z = solve_gambit(game)
    elif method == 2:
        y, z = solve_nashpy(game)

    # Convert to pure policy
    i = np.argmax(y)
    j = np.argmax(z)
    v = game[i][j]
    print (y, z, i, j, v)
    exit()

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


def getCost2go(traveler, occgrid, ugrids, vgrids, egrids, wgrids, verbose = False, iterations = 1, method = 0):

    def haltCost(row, col, target_row, target_col):

        # INF cost if not halting at target
        if  row != target_row or \
            col != target_col:
                return np.inf
        else: # No cost to halt at target
            return 0


    def floodAssignCost(row, col, occgrid, ugrids, vgrids, egrids, wgrids, cost2go, actiongrid, traveler, method = 0):
        stack = set(((row, col),))

        visitedgrid = np.zeros(occgrid.shape)

        while stack:
            row, col = stack.pop()
            errors  = [e[row][col] for e in egrids]
            weights = [w[row][col] for w in wgrids]
            print (errors, weights)

            # If not obstacle..
            if occgrid[row][col] == 0:
                # Assign
                if row == traveler["target"][0] and \
                   col == traveler["target"][1]:
                    cost2go[row][col]    = 0
                    actiongrid[row][col] = "H"
                else:
                    g = getGameForCell(row, col, traveler, ugrids, vgrids, errors, weights, cost2go)
                    solution             = solveGame(g, method)
                    cost2go[row][col]    = solution[4]
                    actiongrid[row][col] = traveler["actionspace"][solution[2]]
                visitedgrid[row][col] = 1

                if row > 0:
                    if visitedgrid[row - 1][col] == 0:
                       stack.add((row - 1, col))
                if row < len(occgrid) - 1:
                    if visitedgrid[row + 1][col] == 0:
                        stack.add((row + 1, col))
                if col > 0:
                    if visitedgrid[row][col - 1] == 0:
                        stack.add((row, col - 1))
                if col < len(occgrid[0]) - 1:
                    if visitedgrid[row][col + 1] == 0:
                        stack.add((row, col + 1))

    m, n = occgrid.shape

    cost2go = np.zeros(occgrid.shape) + 100000000000
    actiongrid = [["x" for col in range(n)] for row in range(m)]

    for i in range(iterations):
        if verbose == True:
            print("iteration: " + str(i + 1) + " / " + str(iterations))
        floodAssignCost(traveler["target"][0], traveler["target"][1], occgrid,
               ugrids, vgrids, egrids, wgrids, cost2go, actiongrid, traveler, method = method)

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




