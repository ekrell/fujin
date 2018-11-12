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



    # Solve a single cell (without considering path)
    solver_tools.getGameForCell(250, 250, traveler, ugrids, vgrids, settings["errors"], settings["weights"])





if __name__ == "__main__":
    main()
