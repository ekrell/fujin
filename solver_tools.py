import travel_tools
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
    workc = work
    # Dynamic programming: include cost2go
    if cost2go is not None and row is not None and col is not None:

        if move == "^":
            if row > 0:
                workc = work + cost2go[row - 1][col]
            else:
                workc = work + 100000000000
        if move == "v":
            if row < len(cost2go) - 1:
                workc = work + cost2go[row + 1][col]
            else:
                workc = work + 100000000000
        if move == "<":
            if col > 0:
                workc = work + cost2go[row][col - 1]
            else:
                workc = work + 100000000000
        if move == ">":
            if col < len(cost2go[0]) - 1:
                workc = work + cost2go[row][col + 1]
            else:
                workc = work + 100000000000
        if move == "*":
            workc = 100000000000

    return workc, work

def getGameForCell(row, col, traveler, ugrids, vgrids, errors, weights, cost2go = None):

    world_actionspace = getWorldActionsForCell(row, col, ugrids, vgrids, errors)

    game = [[0 for wa in range(world_actionspace["num"])] for ta in traveler["actionspace"]]
    game_work = [[0 for wa in range(world_actionspace["num"])] for ta in traveler["actionspace"]]

    for r in range(len(game)):
        for c in range(len(game[0])):
            game[r][c], game_work[r][c] = getOutcome(traveler["actionspace"][r],
                    world_actionspace["uactions"][c], world_actionspace["vactions"][c],
                    weights, traveler, cost2go, row, col)

    game      = np.array(game)
    game_work = np.array(game_work)

    R = [201, 333, 466, 599]
    C = [201, 333, 466, 599]
    if row in R and col in C:
        np.savetxt("test/games/game_4way_res50_scale100_" + str(row) + "-" + str(col) + ".txt", game)

    return game, game_work


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

    return (y, z, i, j, v)


def printGame(game):
    for row in range(len(game)):
        for col in range(len(game[0])):
            print("{0:.2f}".format(game[row][col])),
        print ("")

def printSolution(solution, name = ""):
    print("Nash Solution " + name)
    print("Player 1:" )
    print("    Pure security policy: {}".format(solution[2]))
    print("    Security policy: ")
    print("    "),
    for num in solution[0]:
        print("{} ".format(num)),
    print("")
    print("Player 2:" )
    print("    Pure security policy: {}".format(solution[3]))
    print("    Security policy: ")
    print("    "),
    for num in solution[1]:
        print("{} ".format(num)),
    print("")
    print("Security level: {}".format(solution[4]))


def getCost2go(traveler, occgrid, ugrids, vgrids, egrids, wgrids,
        bounds = None, verbose = False, iterations = None, method = 0):

    # Destination position
    G = (traveler["target"][0], traveler["target"][1])

    # Environment size
    full_m, full_n = occgrid.shape
    m = abs(bounds["lowerright"][0] - bounds["upperleft"][0])
    n = abs(bounds["lowerright"][1] - bounds["upperleft"][1])
    M = m * n # Num cells

    # Minimum distance between neighbors
    d_min = 0
    # Maximum distance between neighbors
    d_max = sqrt(2) * 10 # Put 1000 but unsure max work
    # Maximum cost
    D_max = M * d_max  # D_max must be > (M - 1) * d_max

    if iterations == None:
        iterations = M

    # Get neighbors
    def getNeighbors(i, m, n, env, obstacleFlag = 1):


        B = []

        upAllowed        = False
        downAllowed      = False
        leftAllowed      = False
        rightAllowed     = False
        upleftAllowed    = False
        uprightAllowed   = False
        downleftAllowed  = False
        downrightAllowed = False

        # Up
        if(i[0] - 1 >= 0):
            if(env[i[0] - 1][i[1]] != obstacleFlag):
                upAllowed = True
                B.append((i[0] - 1, i[1], "^"))
        # Down
        if(i[0] + 1 < m):
            if(env[i[0] + 1][i[1]] != obstacleFlag):
                downAllowed = True
                B.append((i[0] + 1, i[1], "v"))
        # Left
        if(i[1] - 1 >= 0):
            if(env[i[0]][i[1] - 1] != obstacleFlag):
                leftAllowed = True
                B.append((i[0], i[1] - 1, "<"))
        # Right
        if(i[1] + 1 < n):
            if(env[i[0]][i[1] + 1] != obstacleFlag):
                rightAllowed = True
                B.append((i[0], i[1] + 1, ">"))
        # Up-Left
        if(i[0] - 1 >= 0 and i[1] - 1 >= 0):
            if(env[i[0] - 1][i[1] - 1] != obstacleFlag):
                upleftAllowed = True
                B.append((i[0] - 1, i[1] - 1, "a"))
        # Up-Right
        if(i[0] - 1 >= 0 and i[1] + 1 < n):
            if(env[i[0] - 1][i[1] + 1] != obstacleFlag):
                uprightAllowed = True
                B.append((i[0] - 1, i[1] + 1, "b"))
        # Down-Left
        if(i[0] + 1 < m and i[1] - 1 >= 0):
            if(env[i[0] + 1][i[1] - 1] != obstacleFlag):
                downleftAllowed = True
                B.append((i[0] + 1, i[1] - 1, "c"))
        # Down-Right
        if(i[0] + 1 < m and i[1] + 1 < n):
            if(env[i[0] + 1][i[1] + 1] != obstacleFlag):
                downrightAllowed = True
                B.append((i[0] + 1, i[1] + 1, "d"))

        # Up-Up-Left
        if(i[0] - 2 >= 0 and i[1] - 1 >= 0 and upAllowed and upleftAllowed and leftAllowed):
            if(env[i[0] - 2][i[1] - 1] != obstacleFlag):
                B.append((i[0] - 2, i[1] - 1, "m"))
        # Up-Up-Right
        if(i[0] - 2 >= 0 and i[1] + 1 < n and upAllowed and uprightAllowed and rightAllowed):
            if(env[i[0] - 2][i[1] + 1] != obstacleFlag):
                B.append((i[0] - 2, i[1] + 1, "n"))
        # Up-Left-Left
        if(i[0] - 1 >= 0 and i[1] - 2 >= 0 and upAllowed and upleftAllowed and leftAllowed):
            if(env[i[0] - 1][i[1] - 2] != obstacleFlag):
                B.append((i[0] - 1, i[1] - 2, "o"))
        # Up-Right-Right
        if(i[0] - 1 >= 0 and i[1] + 2 < n and upAllowed and uprightAllowed and rightAllowed):
            if(env[i[0] - 1][i[1] + 2] != obstacleFlag):
                B.append((i[0] - 1, i[1] + 2, "p"))
        # Down-Down-Left
        if(i[0] + 2 < m and i[1] - 1 >= 0 and downAllowed and downleftAllowed and leftAllowed):
            if(env[i[0] + 2][i[1] - 1] != obstacleFlag):
                B.append((i[0] + 2, i[1] - 1, "w"))
        # Down-Down-Right
        if(i[0] + 2 < m and i[1] + 1 < n and downAllowed and downrightAllowed and rightAllowed):
            if(env[i[0] + 2][i[1] + 1] != obstacleFlag):
                B.append((i[0] + 2, i[1] + 1, "x"))
        # Down-Left-Left
        if(i[0] + 1 < m and i[1] - 2 >= 0 and downAllowed and downleftAllowed and leftAllowed):
            if(env[i[0] + 1][i[1] - 2] != obstacleFlag):
                B.append((i[0] + 1, i[1] - 2, "y"))
        # Down-Right-Right
        if(i[0] + 1 < m and i[1] + 2 < n and downAllowed and downrightAllowed and rightAllowed):
            if(env[i[0] + 1][i[1] + 2] != obstacleFlag):
                B.append((i[0] + 1, i[1] + 2, "z"))


	return B

    # Distance between location i and j
    def distance(i, j):
        dist = pow((pow(i[0] - j[0], 2) + pow(i[1] - j[1], 2)), 0.5)
        return dist

    def f(i, m, n, cost2go, D_max, env):
        B = getNeighbors(i, m, n, env)

        cost_min = D_max
        b_min = (None, None, "-")
        for b in B:
            cost = distance(i, b) + cost2go[b[0]][b[1]]
            if cost < cost_min:
                cost_min = cost
                b_min = b

        return cost_min, b_min

    def getNewCost(i, G, occgrid, m, n, cost2go, D_max):
        cost = D_max
        action = " "

        # Check if target location
        if i == G:
            cost = 0

        # Check if obstacle
        elif occgrid[i[0]][i[1]] == 1:
            cost = D_max

        # Else, calculate cost normally
        else:
            c, b = f(i, occgrid.shape[0], occgrid.shape[1], cost2go, D_max, occgrid)
            cost = min(D_max, c)
            action = b[2]

        return cost, action

    cost2go = np.zeros((full_m, full_n)) + D_max
    work2go = np.zeros((full_m, full_n))
    action2go = np.array([["-" for col in range(full_n)] for row in range(full_m)])
    history = None

    for k in range(iterations):
        for row in range(bounds["upperleft"][0], bounds["lowerright"][0]):
            for col in range(bounds["upperleft"][1], bounds["lowerright"][1]):
                i = (row, col)
                cost2go[i[0]][i[1]], action2go[i[0]][i[1]] = \
                    getNewCost(i, G, occgrid, m, n, cost2go, D_max)

        print (k)

    return cost2go, work2go, action2go, history

def getCost2go__OLD(traveler, occgrid, ugrids, vgrids, egrids, wgrids, bounds = None, verbose = False, iterations = 1, method = 0):

    def haltCost(row, col, target_row, target_col):

        # INF cost if not halting at target
        if  row != target_row or \
            col != target_col:
                return np.inf
        else: # No cost to halt at target
            return 0


    def floodAssignCost(row, col, occgrid, ugrids, vgrids, egrids, wgrids, cost2go, work2go, actiongrid, traveler, bounds = None, method = 0):

        if bounds["upperleft"] == None or bounds["lowerright"] == None:
            bounds = { "upperleft"  : (0, 0),
                       "lowerright" : (occgrid.shape[0], occgrid.shape[1]),
                     }

        stack = set(((row, col),))

        visitedgrid = np.zeros(occgrid.shape)

        while stack:
            row, col = stack.pop()
            errors  = [e[row][col] for e in egrids]
            weights = [w[row][col] for w in wgrids]

            # If not obstacle..
            if occgrid[row][col] == 0:
                # Assign
                if row == traveler["target"][0] and \
                   col == traveler["target"][1]:
                    cost2go[row][col]    = 0
                    actiongrid[row][col] = "*"
                else:
                    g, g_work = getGameForCell(row, col, traveler, ugrids, vgrids,
                                                         errors, weights, cost2go)
                    solution             = solveGame(g, method)
                    cost2go[row][col]    = solution[4]
                    work2go[row][col]    = g_work[solution[2], solution[3]]
                    actiongrid[row][col] = traveler["actionspace"][solution[2]]
                visitedgrid[row][col] = 1

                if row > bounds["upperleft"][0]:
                    if visitedgrid[row - 1][col] == 0:
                       stack.add((row - 1, col))
                if row < bounds["lowerright"][0]:
                    if visitedgrid[row + 1][col] == 0:
                        stack.add((row + 1, col))
                if col > bounds["upperleft"][1]:
                    if visitedgrid[row][col - 1] == 0:
                        stack.add((row, col - 1))
                if col < bounds["lowerright"][1]:
                    if visitedgrid[row][col + 1] == 0:
                        stack.add((row, col + 1))

    m, n = occgrid.shape

    cost2go = np.zeros(occgrid.shape) + 100000000000
    work2go = np.zeros(occgrid.shape)
    actiongrid = np.array([[" " for col in range(n)] for row in range(m)])
    for row in range(len(occgrid)):
        for col in range(len(occgrid[0])):
            if occgrid[row][col] == 1:
                actiongrid[row][col] = '-'
    history = [{"statPath" : None, "statChange" : None} for i in range(iterations)]

    cost2go_prev = np.array(cost2go)

    for i in range(iterations):
        floodAssignCost(traveler["target"][0], traveler["target"][1], occgrid,
            ugrids, vgrids, egrids, wgrids, cost2go, work2go, actiongrid, traveler,
            bounds = bounds, method = method)

        # Calc change in cost2go
        cost2go_diff = np.abs(cost2go - cost2go_prev)
        avg          = np.mean(cost2go)
        avg_diff     = np.mean(cost2go_diff)
        cost2go_prev = np.array(cost2go)

        # Follow path using cost2go
        trace, waypoints = travel_tools.followPath(traveler["start"], actiongrid)
        stat = travel_tools.statPath(trace, waypoints, cost2go, work2go)

        history[i]["statPath"]   = stat
        history[i]["statChange"] = {"avg" : avg, "avg_diff" : avg_diff}

        if verbose == True:
            print("iteration: " + str(i + 1) + " / " + str(iterations))
            print("  Avg cost2go: %f" % (avg))
            print("  Avg cost2go change: %f" % (avg_diff))
            print("  Path: (%d, %d) -> (%d, %d)" % \
                 (traveler["start"][0],  traveler["start"][1],
                 traveler["target"][0], traveler["target"][1]))
            travel_tools.printStatPath(stat, copious = False)

    return cost2go, work2go, actiongrid, history

def writeActiongrid(actiongrid, actionfile, bounds = None):
    numrows = len(actiongrid)
    numcols = len(actiongrid[0])

    if bounds == None:
        bounds = { "upperleft"  : (0, 0),
                   "lowerright" : (actiongrid.shape[0], actiongrid.shape[1]),
                 }

    af = open(actionfile, 'w')
    for row in range(numrows):
        for col in range(numcols):


            af.write("%s" % actiongrid[row][col])


        af.write("\n")
    af.close()

def readActiongrid(actionfile):
    actiongrid = np.genfromtxt(actionfile, delimiter = 1, dtype = "string")
    return actiongrid

