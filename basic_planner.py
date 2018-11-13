import env_setup
import solver_tools
import numpy as np

def main():

    options, args, settings = env_setup.parseOptions()
    speed_cps = 10 # 10 cells per second

    traveler = env_setup.getTraveler(settings["start"], settings["target"], speed_cps, "8way")

    env_setup.printEnv(settings)

    occgrid = env_setup.getOccupancyGrid(settings["occupancy"])

    ugrids, vgrids = env_setup.getVectorGrids(settings["ucomponents"], settings["vcomponents"])

    # Solve a single cell
    r, c = 50, 50
    g = solver_tools.getGameForCell(r, c, traveler, ugrids, vgrids, settings["errors"], settings["weights"])
    solver_tools.printGame(g)
    solution_williams = solver_tools.solveGame(g, 0)
    solution_gambit   = solver_tools.solveGame(g, 1)
    solver_tools.printSolution(solution_williams, "williams")
    solver_tools.printSolution(solution_gambit, "gambit")

    # Solve each cell -> generate nash grid
    nashgrid = solver_tools.getNashGrid(traveler, occgrid, ugrids, vgrids,
                                  settings["errors"], settings["weights"])

    # Save nash grid
    np.savetxt(settings["files"]["nashgrid"], nashgrid)



    # Assign costs -> generate cost2go
    cost2go = solver_tools.getCost2go(traveler, occgrid, ugrids, vgrids,
                                settings["errors"], settings["weights"])

    # Save cost2go
    np.savetxt(settings["files"]["cost2go"], cost2go)


if __name__ == "__main__":
    main()
