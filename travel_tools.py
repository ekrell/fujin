import numpy as np

def move(loc, act):
    if   act == "^":
        return (loc[0] - 1, loc[1])
    elif act == "v":
        return (loc[0] + 1, loc[1])
    elif act == "<":
        return (loc[0],     loc[1] - 1)
    elif act == ">":
        return (loc[0],     loc[1] + 1)
    elif act == "a":
        return (loc[0] - 1, loc[1] - 1)
    elif act == "b":
        return (loc[0] - 1, loc[1] + 1)
    elif act == "c":
        return (loc[0] + 1, loc[1] - 1)
    elif act == "d":
        return (loc[0] + 1, loc[1] + 1)
    elif act == "m":
        return (loc[0] - 2, loc[1] - 1)
    elif act == "n":
        return (loc[0] - 2, loc[1] + 1)
    elif act == "o":
        return (loc[0] - 1, loc[1] - 2)
    elif act == "p":
        return (loc[0] - 1, loc[1] + 2)
    elif act == "w":
        return (loc[0] + 2, loc[1] - 1)
    elif act == "x":
        return (loc[0] + 2, loc[1] + 1)
    elif act == "y":
        return (loc[0] + 1, loc[1] - 2)
    elif act == "z":
        return (loc[0] + 1, loc[1] + 2)
    else:
        return loc

def followPath(start, actiongrid):

    # Returns:
    #   trace: all cells
    #   waypoints: only where turns happen

    loc = start
    act = actiongrid[loc[0]][loc[1]]

    acts      = [act]
    trace     = [loc]
    waypoints = [loc]

    while actiongrid[loc[0]][loc[1]] != "*":
        loc = move(loc, act)
        act = actiongrid[loc[0]][loc[1]]
        acts.append(act)
        trace.append(loc)
        if act != acts[len(acts) - 2]:
            waypoints.append(loc)

    return trace, waypoints

def statPath(trace, waypoints, cost2go, work2go):

    pstat = { "distances"     : None,
              "distance_sum"  : None,
              "num_cells"     : len(trace),
              "num_waypoints" : len(waypoints),
              "costs"         : None,
              "cost_sum"      : None,
              "works"         : None,
              "work_sum"      : None,
    }

    def euclidean(wi, wj):
        return pow(((wi[0] - wj[0]) * (wi[0] - wj[0]) + \
                   (wi[1] - wj[1]) * (wi[1] - wj[1])), 0.5)

    distances = [0 for i in range(len(waypoints) - 1)]
    for i in range(len(waypoints) - 1):
        distances[i] = euclidean(waypoints[i], waypoints[i + 1])

    costs = [0 for i in range(len(trace))]
    works = [0 for i in range(len(trace))]

    for i in range(len(trace)):
        costs[i] = cost2go[trace[i]]
        works[i] = work2go[trace[i]]

    pstat["distances"]    = distances
    pstat["distance_sum"] = sum(distances)
    pstat["costs"]        = costs
    pstat["cost_sum"]    = sum(costs)
    pstat["works"]        = works
    pstat["work_sum"]     = sum(works)

    return pstat


def printStatPath(pstat, copious = True):
    if copious == True:
        print(pstat)
    else:
        print("Path stat:")
        print("  # waypoints: %d" % (pstat["num_waypoints"]))
        print("  # cells:     %d" % (pstat["num_cells"]))
        print("  distance:    %d" % (pstat["distance_sum"]))
        print("  work:        %d" % (pstat["work_sum"]))
        print("  cost:        %d" % (pstat["cost_sum"]))
        print("--------")

    return

