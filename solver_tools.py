import travel_tools
from math import acos, cos, sin, ceil, sqrt, atan2
import numpy as np
import itertools
import time

def getNeighbors(i, m, n, env, occupancyFlag = 1):
    '''Get all valid cells in a cells neighborhood.

    Args:
        i (int, int): Row, column coordinates of cell in environment.
        m (int): Number of rows in environment.
        n (int): number of columns in environment.
        env (2D list of int): Occupancy grid.
        occupancyFlag (int): Flag that indicates occupancy.

    Returns:
        List of (int, int, str): Neighbors as (row, col, action to reach).

    '''

    B = [] # Initialize list of neighbors

    # Diagonals may require checking muliple locations feasibility.
    # Use these booleans to avoid repeated checks.
    upAllowed        = False
    downAllowed      = False
    leftAllowed      = False
    rightAllowed     = False
    upleftAllowed    = False
    uprightAllowed   = False
    downleftAllowed  = False
    downrightAllowed = False

    # Check the neighbors and append to B if within bounds and feasible.
    # Up
    if(i[0] - 1 >= 0):
        if(env[i[0] - 1][i[1]] != occupancyFlag):
            upAllowed = True
            B.append((i[0] - 1, i[1], "^"))
    # Down
    if(i[0] + 1 < m):
        if(env[i[0] + 1][i[1]] != occupancyFlag):
            downAllowed = True
            B.append((i[0] + 1, i[1], "v"))
    # Left
    if(i[1] - 1 >= 0):
        if(env[i[0]][i[1] - 1] != occupancyFlag):
            leftAllowed = True
            B.append((i[0], i[1] - 1, "<"))
    # Right
    if(i[1] + 1 < n):
        if(env[i[0]][i[1] + 1] != occupancyFlag):
            rightAllowed = True
            B.append((i[0], i[1] + 1, ">"))
    # Up-Left
    if(i[0] - 1 >= 0 and i[1] - 1 >= 0):
        if(env[i[0] - 1][i[1] - 1] != occupancyFlag):
            upleftAllowed = True
            B.append((i[0] - 1, i[1] - 1, "a"))
    # Up-Right
    if(i[0] - 1 >= 0 and i[1] + 1 < n):
        if(env[i[0] - 1][i[1] + 1] != occupancyFlag):
            uprightAllowed = True
            B.append((i[0] - 1, i[1] + 1, "b"))
    # Down-Left
    if(i[0] + 1 < m and i[1] - 1 >= 0):
        if(env[i[0] + 1][i[1] - 1] != occupancyFlag):
            downleftAllowed = True
            B.append((i[0] + 1, i[1] - 1, "c"))
    # Down-Right
    if(i[0] + 1 < m and i[1] + 1 < n):
        if(env[i[0] + 1][i[1] + 1] != occupancyFlag):
            downrightAllowed = True
            B.append((i[0] + 1, i[1] + 1, "d"))
    # Up-Up-Left
    if(i[0] - 2 >= 0 and i[1] - 1 >= 0 and upAllowed \
                     and upleftAllowed and leftAllowed):
        if(env[i[0] - 2][i[1] - 1] != occupancyFlag):
            B.append((i[0] - 2, i[1] - 1, "m"))
    # Up-Up-Right
    if(i[0] - 2 >= 0 and i[1] + 1 < n and upAllowed \
                     and uprightAllowed and rightAllowed):
        if(env[i[0] - 2][i[1] + 1] != occupancyFlag):
            B.append((i[0] - 2, i[1] + 1, "n"))
    # Up-Left-Left
    if(i[0] - 1 >= 0 and i[1] - 2 >= 0 and upAllowed \
                     and upleftAllowed and leftAllowed):
        if(env[i[0] - 1][i[1] - 2] != occupancyFlag):
            B.append((i[0] - 1, i[1] - 2, "o"))
    # Up-Right-Right
    if(i[0] - 1 >= 0 and i[1] + 2 < n and upAllowed \
                     and uprightAllowed and rightAllowed):
        if(env[i[0] - 1][i[1] + 2] != occupancyFlag):
            B.append((i[0] - 1, i[1] + 2, "p"))
    # Down-Down-Left
    if(i[0] + 2 < m and i[1] - 1 >= 0 and downAllowed \
                    and downleftAllowed and leftAllowed):
        if(env[i[0] + 2][i[1] - 1] != occupancyFlag):
            B.append((i[0] + 2, i[1] - 1, "w"))
    # Down-Down-Right
    if(i[0] + 2 < m and i[1] + 1 < n and downAllowed \
                    and downrightAllowed and rightAllowed):
        if(env[i[0] + 2][i[1] + 1] != occupancyFlag):
            B.append((i[0] + 2, i[1] + 1, "x"))
    # Down-Left-Left
    if(i[0] + 1 < m and i[1] - 2 >= 0 and downAllowed \
                    and downleftAllowed and leftAllowed):
        if(env[i[0] + 1][i[1] - 2] != occupancyFlag):
            B.append((i[0] + 1, i[1] - 2, "y"))
    # Down-Right-Right
    if(i[0] + 1 < m and i[1] + 2 < n and downAllowed \
                    and downrightAllowed and rightAllowed):
        if(env[i[0] + 1][i[1] + 2] != occupancyFlag):
            B.append((i[0] + 1, i[1] + 2, "z"))
    return B


def getWorldActionsForCell(row, col, ugrids, vgrids, errors, resolution = 10):
    '''Generate possible discrete actions that the environment may select
       within an error range. Does so for u and v components of force.

    Args:
        row (int): Row in environment.
        col (int): Column in environment.
        ugrids (array(float, ndim=1)): U components of forces.
        vgrids (array(float, ndim=1)): V components of forces.
        errors (array(float, ndim=1)): Error of u, v at that location.
        resolution (int): Number of equally-spaced action intervals.

    Returns:
        A dict with the following keys:
            uactions (array(float, ndim=1)): Possible u components.
            vactions (array(float, ndim=1)): Possible v components.
            num (int): Number of actions available across all vectors.
    '''


    numVectors = len(ugrids)
    uranges = [None for i in range(numVectors)]
    vranges = [None for i in range(numVectors)]

    # Number of available actions is combination of all
    # action choices for all vectors
    numActions = 1

    # For each vector,
    for g in range(numVectors):
        uranges[g] = np.linspace(ugrids[g][row][col] - errors[g], ugrids[g][row][col] + errors[g], resolution)
        vranges[g] = np.linspace(vgrids[g][row][col] - errors[g], vgrids[g][row][col] + errors[g], resolution)
        numActions = numActions * len(uranges[g]) # Another combination

    uactionspace = list(itertools.product(*uranges))
    vactionspace = list(itertools.product(*vranges))

    # Store in dictionary
    world_actionspace = { "uactions"   : uactionspace,
                          "vactions"   : vactionspace,
                          "num"        : numActions,
                        }

    return world_actionspace



def getVectorSum(us, vs, weights):
    '''Caclulate weighted sum of arbitrary number of vectors.

    Args:
        us (array(float, ndim=1)): U component of ith vector.
        vs (array(float, ndim=1)): V component of ith vector.
        weights (array(float, ndim=1)): Weight of each vector.

    Returns:
        (tuple): Tuple containing:
            utotal (float): weighted sum of u components
            vtotal (float): weighted sum of v components
    '''

    utotal = 0
    vtotal = 0

    # Assumes us, vs of same length
    numVectors = len(us)

    for i in range(numVectors):
        utotal = utotal + weights[i] * us[i]
        vtotal = vtotal + weights[i] * vs[i]

    return (utotal, vtotal)


def getVectorDiff(us, vs, weights):
    '''Caclulate weighted different between a vector and
        summation arbitrary number of additional vectors.
        The first element of us and vs is the target vector
        that all other vectors subtract.

    Args:
        us (array(float, ndim=1)): U component of ith vector.
        vs (array(float, ndim=1)): V component of ith vector.
        weights (array(float, ndim=1)): Weight of each vector.

    Returns:
        (tuple): tuple containing:
            utotal (float): weighted diff of u components
            vtotal (float): weighted diff of v components
    '''

    utotal = us[0] # Get first vector
    vtotal = vs[0] # Get first vector
    us.pop(0)
    vs.pop(0)

    numVectors = len(us)

    # Subsequent vectors are (weighted) RHS of subtraction
    for i in range(numVectors):
        utotal = utotal - weights[i] * us[i]
        vtotal = vtotal - weights[i] * vs[i]

    return utotal, vtotal


def magdir2uv(mag, dir_radians):
    '''Converts a force in form (magnitude, direction) to
    its vector components (u, v).

    Args:
        mag (float): magnitude of force.
        dir_radians (float): direction of force in radians.

    Returns:
        (tuple): tuple containing:
            u (float): u component.
            v (float): v component.
    '''

    u = mag * cos(dir_radians)
    v = mag * sin(dir_radians)
    return u, v

def uv2magdir(u, v):
    '''Converts a force in vector form (u, v) to
    its force magnitude and direction.

    Args:
        u (float): u component
        v (float): v component

    Returns:
        (tuple): tuple containing:
            m (float): magnitude.
            d (float): direction in radians.
    '''

    m = sqrt(u * u + v * v) # Calculate magnitude
    d = atan2(v, u) # Calculate direction (radians)
    return m, d

def getOutcome(move, us, vs, weights, traveler, D_max,
        cost2go = None, row = None, col = None):
    '''Calculate the cost of a particular action choice.
        By default, only gives the cost of performing a particular
        action given the force that apply to the traveler.
        If cost2go, row, and col specified, adds the cost of
        the reaching that location. This is related to dynamic programming,
        where the cost of an action includes the cost of subsequent actions.

    Args:
        move (string): Symbol that represents move, such as "^" : "up".
        us (array(float, ndim=1)): U components of forces applied to traveler.
        vs (array(float, ndim=1)): V components of forces applied to traveler.
        weights (array(float, ndim=1)): Relative importance of each vector.
        traveler (dict of 'Traveler'): See DEVELOPERS.md for data structures.
        D_max (float): Largest possible cost, or greater.
        cost2go (array_like(float, ndim=2)): Cost of the each region location.
            Defaults to None.
        row (int): Index into first dimension of cos2go. Defaults to None.
        col (int): Index into second dimension of cost2go. Defaults to None.

    Returns:
        (tuple): tuple containing:
            work (float): cost based on work.
            workc (float): cost based on work and subsequent location.
    '''

    # Get resultant vector of all weighted world sources
    uw, vw = getVectorSum(us, vs, weights)

    # Break traveler magnitude, direction into vector
    ut, vt = magdir2uv(traveler["speed_cps"], traveler["action2radians"][move])

    # Get vector difference (required applied force)
    ua, va = getVectorDiff([uw, ut], [vw, vt], [1, 1])

    # Combine applied u, v to get mag, dir
    maga, dira = uv2magdir(ua, va)

    # distance
    distance = 0
    if move == "^" or move == "v" or move == "<" or move == ">":
        distance = 1
    if move == "a" or move == "b" or move == "c" or move == "d":
        distance = 1.41421
    if move == "m" or move == "n" or move == "o" or move == "p" or \
       move == "w" or move == "x" or move == "y" or move == "z":
        distance = 2.236

    # Calculate work
    work = maga * distance
    #work = distance * ua + distance * va
    workc = work
    # Dynamic programming: include cost2go
    if cost2go is not None and row is not None and col is not None:

        m = len(cost2go)
        n = len(cost2go[0])

        if move == "^":
            workc = work + cost2go[row - 1][col]
        elif move == "v":
            workc = work + cost2go[row + 1][col]
        elif move == "<":
            workc = work + cost2go[row][col - 1]
        elif move == ">":
            workc = work + cost2go[row][col + 1]
        elif move == "a":
            workc = work + cost2go[row - 1][col - 1]
        elif move == "b":
            workc = work + cost2go[row - 1][col + 1]
        elif move == "c":
            workc = work + cost2go[row + 1][col - 1]
        elif move == "d":
            workc = work + cost2go[row + 1][col + 1]
        elif move == "m":
            workc = work + cost2go[row - 2][col - 1]
        elif move == "n":
            workc = work + cost2go[row - 2][col + 1]
        elif move == "o":
            workc = work + cost2go[row - 1][col - 2]
        elif move == "p":
            workc = work + cost2go[row - 1][col + 2]
        elif move == "w":
            workc = work + cost2go[row + 2][col - 1]
        elif move == "x":
            workc = work + cost2go[row + 2][col + 1]
        elif move == "y":
            workc = work + cost2go[row + 1][col - 2]
        elif move == "z":
             workc = work + cost2go[row + 1][col + 2]
        elif move == "*":
                workc = d_max
        else:
            print("Unknown move")

    return workc, work

def getGameForCell(row, col, traveler, ugrids, vgrids,
                   errors, weights, env, D_max, cost2go = None):
    '''Generates a 2-player, 0-sum game for a cell in the region.
        Rows are traveler actions and columns are world actions.
        Traver actions based on its movement scheme.
        World actions based on changing force components within error.
        Costs are the work by traveler to maintain target velocity given
        world forces. If cost2go is supplied, cost includes the
        subsequent cost of where traveler moves to taking the action.
        This is for the dynamic programming algorithm.

    Args:
        row (int): Location of cell.
        col (int): Location of cell.
        traveler (dict of 'traveler'): See DEVELOPERS.md for data structures.
        ugrids (List of array(float, ndim=2)): U components of forces.
        vgrids (List of array(float, ndim=2)): V components of forces.
        errors (List of array(float, ndim=2)): Errors of forces.
        weights (List of array(float, ndim=2)): Weights of forces.
        env (array(float, ndim=2)): Occupancy grid of region.
        D_max (float): Largest possible cost, or greater.
        cost2go (array(float, ndim=2)): cost2go of region cells.
            Defaults to None.

    Returns:
        (tuple): tuple containing:
            game (array(float, ndim=2)): 2-player 0-sum game.
            game_work (array(float, ndim=2)): work only for actions.
    '''

    # All actions available to world
    world_actionspace = getWorldActionsForCell(row, col, ugrids, vgrids,
                                               errors)

    game = [[0 for wa in range(world_actionspace["num"])] for ta \
            in traveler["actionspace"]]
    game_work = [[0 for wa in range(world_actionspace["num"])] for ta \
            in traveler["actionspace"]]

    B = getNeighbors((row, col), env.shape[0], env.shape[1], env)
    Bmove = [b[2] for b in B]

    for r in range(len(game)):
        Bvalid = False
        if traveler["actionspace"][r] in Bmove:
            Bvalid = True
        for c in range(len(game[0])):
            if Bvalid:
                game[r][c], game_work[r][c] = getOutcome(traveler["actionspace"][r],
                        world_actionspace["uactions"][c], world_actionspace["vactions"][c],
                        weights, traveler, D_max, cost2go, row, col)
            else: # If invalid move, do not bother with work
                game[r][c], game_work[r][c] = D_max, D_max

    game      = np.array(game)
    game_work = np.array(game_work)
    return game, game_work


def solve_williams(payoff_matrix, iterations = 100):
    '''Approximate solver for 2-player, 0-sum matrix game Nash equilibrium.
        Simulated a sequence of alternating plays where players respond
        to strategy of other players. After a set number of iterations,
        terminates. The proportion of each selection is the probability
        in the mixed policy.

    References:
        http://code.activestate.com/recipes/496825-game-theory-payoff-matrix-solver/

    Args:
        payoff_matrix (array(float, ndim=2)): Game to solve.
        iterations (int): Number of solver iterations. Defaults to 100.

    Returns:
        (tuple): tuple containing:
            array(float, ndim=1): mixed policy for row player.
            array(float, ndim=1): mixed policy for column player.
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

def solveGame(game, method = 0):
    '''Solves a 2-player, 0-sum matrix game using selected method.
        Currently only a single method is available.

    Args:
        game (array(float, ndim=2)): Matrix game to solve.
        method (int): Numeric flag to choose game.

    Returns:
        (tuple): tuple containing:
            y (array(float, ndim=1)): mixed policy for row player.
            z (array(float, ndim=1)): mixed policy for column player.
            i (int): pure policy for row player.
            j (int): pure policy for column player.
            v (float): value of game when applying pure policies.
    '''

    # Call selected method
    if method == 0:
        y, z = solve_williams(game)

    # Convert to pure policy
    i = np.argmax(y)
    j = np.argmax(z)
    v = game[i][j]

    return (y, z, i, j, v)


def printGame(game):
    '''Prints a matrix game.

    Args:
        game (array(float, ndim=2)): 2-player matrix game.

    Returns:
        None.
    '''

    for row in range(len(game)):
        for col in range(len(game[0])):
            print("{0:.2f}".format(game[row][col])),
        print ("")

def printSolution(solution, name = ""):
    '''Prints solution of 2-player, 0-sum matrix game.
        Includes the original mixed policies as well
        as pure policies that are the highest-probability mixed option.
        An optional name can be provided to describe the solution source.

    Args:
        solution (tuple): tuple containing:
            array(float, ndim=1): mixed policy for row player.
            array(float, ndim=1): mixed policy for column player.
            int: pure policy for row player.
            int: pure policy for column player.
            float: value of game when applying pure policies.
        name (string): Solution identification. Defaults to "".

    Returns:
        None.
    '''

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
               bounds = None, verbose = False, iterations = None, method = 0,
               style = "fast"):
    '''Builds the dynamic programming cost2go value for each cell in the region.
        Does so iteratively, since incorrect values may be temporarily recorded.

    See 'Game-theoretic Dynamic Programming for Motion Planning of Unmanned
        Surface Vehicles Given Uncertain Environmental Forces' - Evan Krell,
        Luis Rodolfo GARCIA CARRILLO, Scott A. King (Under Review).

    Args:
        traveler (dict of 'traveler'): see developers.md for data structs.
        occgrid (array(int, ndim=2)): Binary occupancy grid.
        ugrids (List of array(float, ndim=2)): U components of forces.
        vgrids (List of array(float, ndim=2)): V components of forces.
        egrids (List of array(float, ndim=2)): Error of u, v at that location.
        wgrids (List of array(float, ndim=2)): Weights of forces.
        bounds (dict): A dict with the following keys:
            upperleft (tuple): A tuple containing:
                (int): Row of upper-left corner of boundary to solve.
                (int): Column of upper-left corner of boundary to solve.
            lowerright (tuple): A tuple containing:
                (int): Row of lower-right corner of boundary to solve.
                (int): Column of lower-right corner of boundary to solve.
            Defaults to None, meaning the entire region is within bounds.
        verbose (bool): Whether to print summary after every iteration.
            Defaults to False.
        iterations (integer). Number of iterations of dynamic programming.
            Defaults to None.
        method (integer). Selects a matrix game solver.
            See README.md for choices. Defaults to 0.
        style (integer). Which algorithm for updating cost2go values.
            "fast" or "slow" supported. Defaults to "fast".

    Returns:
        cost2go (array(float, ndim=2): Cost of visiting each cell in region.
        work2go (array(float, ndim=2): Work done at each cell in region.
        action2go (array(string, ndim=2): Motion plan over region as symbols.
        uenvgrid (array(float, ndim=2): U of force applied by environment.
        venvgrid (array(float, ndim=2): V of force applied by environment.
        history (List of dict): A dictionary with the following keys:
            statPath (dict of 'statPath'): DEVELOPMENT.md for data structs.
            statChange (dict of 'statChange'): DEVELOPEMENT.md for data structs.
    '''

    # If no bounds set, set bounds to region extent
    if bounds["upperleft"] is None:
        bounds["upperleft"] = (0, 0)
    if bounds["lowerright"] is None:
        bounds["lowerright"] = (occgrid.shape[0] - 1, occgrid.shape[1] - 1)

    # Destination position
    G = (traveler["target"][0], traveler["target"][1])

    # Environment size
    full_m, full_n = occgrid.shape
    m = abs(bounds["lowerright"][0] - bounds["upperleft"][0]) + 1
    n = abs(bounds["lowerright"][1] - bounds["upperleft"][1]) + 1
    M = m * n # Num cells

    # Minimum distance between neighbors
    d_min = 0
    # Maximum distance between neighbors
    d_max = sqrt(2) * 10 # Put 10 but unsure max work
    # Maximum cost
    D_max = M * d_max # Since costs accum with size of region.

    if iterations == None:
        iterations = M

    def distance(i, j):
        '''Euclidean distance between points i and j.

        Args:
            i (tuple): tuple containing:
                (int): row of ith point.
                (int): column of ith point.
            j (tuple): tuple containing:
                (int): row of jth point.
                (int): column of jth point.

        Returns:
            dist (float): Euclidean distance.

        '''

        dist = pow((pow(i[0] - j[0], 2) + pow(i[1] - j[1], 2)), 0.5)
        return dist

    ##def f(i, m, n, cost2go, D_max, env):
    ##    B = getNeighbors(i, m, n, env)

    ##    cost_min = D_max
    ##    b_min = (None, None, "-")
    ##    for b in B:
    ##        cost = distance(i, b) + cost2go[b[0]][b[1]]
    ##        if cost < cost_min:
    ##            cost_min = cost
    ##            b_min = b

    ##    return cost_min, b_min

    def f_work(i, m, n, cost2go, D_max, env, traveler, ugrids, vgrids, egrids,
               wgrids, method):
        '''Calculate work, action at cell i.
            Work to maintain traveler's target velocity in presence
            of environment forces. Action is motion plan action to take at i.
            If forces have nonzero errors, solution dependent on solving
            2-player, 0-sum matrix game. Otherwise, calculate with the
            static forces at i.

        Args:
            i (tuple): tuple containing:
                (int): row of ith point.
                (int): column of ith point.
            m (int): Number of rows in region.
            n (int): Number of columns in region.
            cost2go: (array(float, ndim=2): Cost of visiting each cell in region.
            D_max (float): Largest possible cost, or greater.
            env (array(int, ndim=2)): Binary occupancy grid.
            traveler (dict of 'traveler'): see developers.md for data structs.
            ugrids (List of array(float, ndim=2)): U components of forces.
            vgrids (List of array(float, ndim=2)): V components of forces.
            egrids (List of array(float, ndim=2)): Errors of forces.
            wgrids (List of array(float, ndim=2)): Weights of forces.
            method (integer). Selects a matrix game solver.
                See README.md for choices. Defaults to 0.

        Returns:
            (float): cost of selected action.
            (string): selected action as action symbol.
            (float): work to perform selected action.
            uenv (float): U component of summed environment forces.
            venv (float): V component of summed environment forces.
        '''

        errors = [e[i[0]][i[1]] for e in egrids]
        weights = [w[i[0]][i[1]] for w in wgrids]
        g, g_work = getGameForCell(i[0], i[1], traveler, ugrids, vgrids,
                                   errors, weights, env, D_max, cost2go)

        solution = solveGame(g, method)
        world_actionspace = getWorldActionsForCell(i[0], i[1], ugrids,
                                                   vgrids, errors)

        uenv = sum(world_actionspace["uactions"][solution[3]])
        venv = sum(world_actionspace["vactions"][solution[3]])

        return (solution[4], traveler["actionspace"][solution[2]],
                g_work[solution[2], solution[3]], uenv, venv)


    def getNewCost(i, G, occgrid, m, n, cost2go, D_max, traveler,
                   ugrids, vgrids, egrids, wgrids):

        '''Get action and cost at cell i. Only solves game, calculates work
            if appropriate for point i. The goal's cost is always 0 with
            the halt action. Obstacles have action (" ") and maximal cost.

        Args:
            i (tuple): tuple containing:
                (int): row of ith point.
                (int): column of ith point.
            G
            occgrid (array(int, ndim=2)): Binary occupancy grid.
            m (int): Number of rows in region.
            n (int): Number of columns in region.
            cost2go: (array(float, ndim=2): Cost of visiting each cell in region.
            D_max (float): Largest possible cost, or greater.
            traveler (dict of 'traveler'): see developers.md for data structs.
            ugrids (List of array(float, ndim=2)): U components of forces.
            vgrids (List of array(float, ndim=2)): V components of forces.
            egrids (List of array(float, ndim=2)): Errors of forces.
            wgrids (List of array(float, ndim=2)): Weights of forces.

        Returns:
            cost (float): cost of selected action.
            action (string): selected action as action symbol.
            work (float): work to perform selected action.
            uenv (float): U component of summed environment forces.
            venv (float): V component of summed environment forces.
        '''


        cost = D_max
        action = " "
        work = D_max

        uenv = 0
        venv = 0
        for j in range(len(ugrids)):
            uenv = uenv + ugrids[j][i[0]][i[1]]
            venv = venv + vgrids[j][i[0]][i[1]]

        calcCost = True

        # Check if target location
        if i == G:
            cost = 0
            work = 0
            action = "*"
            calcCost = False
        # Check if obstacle
        elif occgrid[i[0]][i[1]] == 1:
            cost = D_max
            work = D_max
            action = " "
            calcCost = False

        # Else, calculate cost normally
        if calcCost:
            c, b, w, u, v = f_work(i, occgrid.shape[0], occgrid.shape[1],
                    cost2go, D_max, occgrid, traveler, ugrids, vgrids, egrids,
                    wgrids, 0)
            if c >= D_max:
                cost = D_max
                action = '-'
            else:
                cost = c
                action = b
                uenv = u
                venv = v

        return cost, action, work, uenv, venv

    cost2go = np.zeros((full_m, full_n)) + D_max
    work2go = np.zeros((full_m, full_n))
    action2go = np.array([["-" for col in range(full_n)] for row in \
            range(full_m)])
    uenvgrid = np.zeros((full_m, full_n))
    venvgrid = np.zeros((full_m, full_n))

    uenvgrid = np.array(ugrids[0])   # DEBUG
    venvgrid = np.array(vgrids[0])   # DEBUG

    history = [{"statPath" : None, "statChange" : None} for i in \
            range(iterations)]

    # Save copy of cost2go as 'prev' to compare to next iter's version
    cost2go_prev = np.array(cost2go)

    for k in range(iterations):

        if style == "fast":
            # Queue of locations to evaluate
            visitQueue = [G]
            # Queue history
            visitHistory = set()
            # Bottoms-up visit of all possible locations
            while visitQueue:
                    i = visitQueue.pop(0)
                    # Assign cost2go at i
                    cost2go[i[0]][i[1]], action2go[i[0]][i[1]],
                    work2go[i[0]][i[1]], uenvgrid[i[0]][i[1]],
                    venvgrid[i[0]][i[1]] = \
                        getNewCost(i, G, occgrid, m, n, cost2go, D_max,
                                traveler, ugrids, vgrids, egrids, wgrids)
                    # Add i's neighbors to list
                    B = getNeighbors(i, occgrid.shape[0], occgrid.shape[1],
                                     occgrid)
                    for b in B:
                        b = (b[0], b[1])
                        if  b[0] >= bounds["upperleft"][0] and b[0] <= \
                            bounds["lowerright"][0] and \
                            b[1] >= bounds["upperleft"][1] and b[1] <= \
                            bounds["lowerright"][1]:
                            if b not in visitHistory:
                                visitQueue.append(b)
                                visitHistory.add(b)
        elif style == "slow":
            for row in range(bounds["upperleft"][0], bounds["lowerright"][0]):
                for col in range(bounds["upperleft"][1],
                                 bounds["lowerright"][1]):
                    i = (row, col)
                    cost2go[i[0]][i[1]], action2go[i[0]][i[1]],
                    work2go[i[0]][i[1]], uenvgrid[i[0]][i[1]],
                    venvgrid[i[0]][i[1]] = \
                        getNewCost(i, G, occgrid, m, n, cost2go, D_max,
                                traveler, ugrids, vgrids, egrids, wgrids)

        # Calc iteration stats
        cost2go_diff = np.abs(cost2go - cost2go_prev)
        avg = np.mean(cost2go)
        avg_diff = np.mean(cost2go_diff)
        cost2go_prev = np.array(cost2go)
        # Follow path using cost2go
        trace, waypoints = travel_tools.followPath(traveler["start"], action2go)
        stat = travel_tools.statPath(trace, waypoints, cost2go, work2go)
        history[k]["statPath"] = stat
        history[k]["statChange"] = {"avg" : avg, "avg_diff" : avg_diff}
        # Print iteration info
        if verbose:
            print("Iteration: " + str(k + 1) + "/" + str(iterations))
            print("  Avg cost2go: %f" % (avg))
            print("  Avg cost2go change: %f" % (avg_diff))
            print("  Path: (%d, %d) -> (%d, %d)" % \
                    (traveler["start"][0],  traveler["start"][1],
                     traveler["target"][0], traveler["target"][1]))
            travel_tools.printStatPath(stat, copious = False)

    return cost2go, work2go, action2go, uenvgrid, venvgrid, history

def writeActiongrid(actiongrid, actionfile, bounds = None):
    '''Writes action grid (motion plan as action symbols) to ASCII file.

    Args:
        actiongrid (array(string, ndim=2): Motion plan over region as symbols.
        actionfile (string): Path to write ASCII file.
        bounds (dict): A dict with the following keys:
            upperleft (tuple): A tuple containing:
                (int): Row of upper-left corner of boundary to solve.
                (int): Column of upper-left corner of boundary to solve.
            lowerright (tuple): A tuple containing:
                (int): Row of lower-right corner of boundary to solve.
                (int): Column of lower-right corner of boundary to solve.
            Defaults to None, meaning the entire region is within bounds.

    Returns:
        None.
    '''

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
    '''Reads action grid (motion plan as action symbols) from ASCII file.

    Args:
        actionfile (string): Path to read ASCII file.

    Returns:
        actiongrid (array(string, ndim=2): Motion plan over region as symbols.
    '''

    actiongrid = np.genfromtxt(actionfile, delimiter = 1, dtype = "string")
    return actiongrid

