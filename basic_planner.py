import env_setup
import solver_tools
import travel_tools
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

def main():

    history = None

    options, args, settings = env_setup.parseOptions()
    speed_cps = 10 # 10 cells per second

    traveler = env_setup.getTraveler(settings["start"], settings["target"], speed_cps, "4way")

    env_setup.printEnv(settings)

    occgrid = env_setup.getOccupancyGrid(settings["occupancy"])
    ugrids, vgrids = env_setup.getVectorGrids(settings["ucomponents"], settings["vcomponents"])
    wgrids = env_setup.getWeightGrids(occgrid, settings["weights"], settings["weightgrids"], len(ugrids))
    egrids  = env_setup.getErrorGrids (occgrid, settings["errors"],  settings["errorgrids"], len(ugrids))

    if settings["reuse"] == False:
        # Assign costs -> generate cost2go
        cost2go, work2go, actiongrid, history = solver_tools.getCost2go(traveler, 
          occgrid, ugrids, vgrids, egrids, wgrids, 
          bounds = settings["bounds"], verbose =  settings["verbose"], 
          method = settings["method"], iterations = settings["iterations"])
        # Save cost2go
        np.savetxt(settings["files"]["cost2go"], cost2go)
        # Save work2go
        np.savetxt(settings["files"]["work2go"], work2go)
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

    # Test a single path
    if traveler["start"] is not None:
        # Generate path to goal
        trace, waypoints = travel_tools.followPath(traveler["start"], actiongrid)
        # Stat path to goal
        stat = travel_tools.statPath(trace, waypoints, cost2go, work2go)
        # Print stat
        travel_tools.printStatPath(stat, copious = False)
        
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
              "col_avg"      : col_avg,
              "col_avg_diff" : col_avg_diff,
            })
        history_df.index.name = "iteration"

    # Save to csv
    if settings["files"]["pandas"] is not None and history is not None:
        history_df.to_csv(settings["files"]["pandas"]) 

    # Plot convergence
    if history is not None:
        plt.scatter(history_df.index.values, history_df["work"])
        plt.show()


if __name__ == "__main__":
    main()
