import env_setup
import solver_tools
import travel_tools
import visual_tools
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

def main():

    #########
    # Setup #
    #########
    # Init blank history
    history = None
    # Parse options
    options, args, settings = env_setup.parseOptions()
    # Create 'traveler' (dict)
    traveler = env_setup.getTraveler(settings["start"], settings["target"],
                                                settings["speed"], "16way")

    env_setup.printEnv(settings)
    occgrid = env_setup.getOccupancyGrid(settings["occupancy"])
    ugrids, vgrids = env_setup.getVectorGrids(settings["ucomponents"],
                                     settings["vcomponents"], occgrid)
    wgrids = env_setup.getWeightGrids(occgrid, settings["weights"],
                              settings["weightgrids"], len(ugrids))
    egrids  = env_setup.getErrorGrids (occgrid, settings["errors"],
                               settings["errorgrids"], len(ugrids))
    ############
    # Planning #
    ############
    if settings["reuse"] == False:
        # Assign costs -> generate cost2go
        cost2go, work2go, actiongrid, uenvgrid, venvgrid, history = solver_tools.getCost2go(traveler,
          occgrid, ugrids, vgrids, egrids, wgrids,
          bounds = settings["bounds"], verbose =  settings["verbose"],
          iterations = settings["iterations"])
        # Save cost2go
        np.savetxt(settings["files"]["cost2go"], cost2go)
        # Save work2go
        np.savetxt(settings["files"]["work2go"], work2go)
        # Save envgrid
        ###np.savetxt(settings["files"]["uenvgrid"], uenvgrid)
        ###np.savetxt(settings["files"]["venvgrid"], venvgrid)
        # Save actiongrid
        solver_tools.writeActiongrid(actiongrid, settings["files"]["actiongrid"])
        # Save convergence history
        if settings["files"]["pickle"] is not None:
            with open(settings["files"]["pickle"], 'wb') as handle:
                pickle.dump(history, handle, protocol = pickle.HIGHEST_PROTOCOL)
    else:
        # Use existing cost2go, work2go, actiongrid
        cost2go    = np.loadtxt(settings["files"]["cost2go"])
        work2go    = np.loadtxt(settings["files"]["work2go"])
        actiongrid = solver_tools.readActiongrid(settings["files"]["actiongrid"])

        if settings["files"]["pickle"] is not None:
            with open(settings["files"]["pickle"], 'rb') as handle:
                history = pickle.load(handle)

    ###############
    # Path Follow #
    ###############
    # Test a single path
    if traveler["start"] is not None:
        # Generate path to goal
        trace, waypoints = travel_tools.followPath(traveler["start"], actiongrid)
        # Stat path to goal
        stat = travel_tools.statPath(trace, waypoints, cost2go, work2go)
        # Print stat
        travel_tools.printStatPath(stat, copious = False)

    ##############
    # Path Stats #
    ##############
    # Generate convergence table
    if history is not None:
        col_num_waypoints = [s["statPath"]["num_waypoints"] for s in history]
        col_distance      = [s["statPath"]["distance_sum"]  for s in history]
        col_work          = [s["statPath"]["work_sum"]      for s in history]
        col_cost          = [s["statPath"]["cost_sum"]      for s in history]
        col_avg           = [s["statChange"]["avg"]         for s in history]
        col_avg_diff      = [s["statChange"]["avg_diff"]    for s in history]
        history_df = pd.DataFrame(
            { "waypoints"    : col_num_waypoints,
              "distance"     : col_distance,
              "work"         : col_work,
              "cost"         : col_cost,
              "avg"          : col_avg,
              "avg_diff"     : col_avg_diff,
            })
        history_df.index.name = "iteration"

    # Save to csv
    if settings["files"]["pandas"] is not None and history is not None:
        history_df.to_csv(settings["files"]["pandas"])

    ##############
    # Path Plots #
    ##############
    # Plot convergence
    if history is not None:
        visual_tools.plotPathStatHistory(history_df, settings["files"]["plots"])
        ax = visual_tools.plotPath(trace, waypoints, settings["occupancy"][0],
                                          occgrid, settings["files"]["plots"])
        ###ax = visual_tools.plotVector(ugrids, vgrids, settings["occupancy"][0],
        ###                                  occgrid, settings["files"]["plots"],
        ###                                  color = "white")
        ax = visual_tools.plotVector([uenvgrid], [venvgrid], settings["occupancy"][0],
                                          occgrid, settings["files"]["plots"],
                                                              color = "white")
    if actiongrid is not None:
        visual_tools.plotActions(actiongrid, traveler["action2radians"], settings["occupancy"][0],
                                                              occgrid, settings["files"]["plots"])

    if settings["files"]["plots"] is not None:
        ax = visual_tools.plotPath(trace, waypoints, settings["occupancy"][0],
                occgrid, settings["files"]["plots"] + "_full", init = False)


if __name__ == "__main__":
    main()
