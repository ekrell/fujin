import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from optparse import OptionParser
import time
import solver_tools

def main():

    # Define options
    parser = OptionParser()
    parser.add_option("-g", "--game",       dest = "game",       metavar = "GAME",
        help = "game as numpy txt")
    parser.add_option("-i", "--iterations", dest = "iterations", metavar = "ITERATIONS",
        default = 100, type = "int",
        help = "number of iterations for approximate Nash solver")
    
    (options, args) = parser.parse_args()
    if options.game is None:
        print("[-] Must specify '--game'")
        exit()

    game = np.loadtxt(options.game)

    startTime = time.time()
    solution = solver_tools.solveGame(game)
    stopTime = time.time()
    elapsedTime = stopTime - startTime

    print("")
    print("Game: %s" % options.game)
    solver_tools.printSolution(solution)
    print("Solve game size (%d,%d) of %d iterations in %.5f secs" % (game.shape[0], game.shape[1], 
                                                                 options.iterations, elapsedTime))
    print("")    

if __name__ == "__main__":
    main()
