import solver_tools
import numpy as np
import time

gameTemplate = { "matrix"     : None,
                 "pure_nash"  : None,
                 "mixed_nash" : None,
                 "name"       : None,
               } 

def checkGame(game, method = 0):
    start_time = time.time()
    y, z, i, j, v = solver_tools.solveGame(game["matrix"], method)
    duration = time.time() - start_time

    print("Game %s" % (game["name"]))
    print("  Known pure Nash:"),
    for idx in range(len(game["pure_nash"])):
        print("(%d, %d), " % (game["pure_nash"][idx][0], game["pure_nash"][idx][1])),
    print("")
    print("  Found pure Nash: (%d, %d)" % (i, j))
    print("  Known mixed Nash:")
    print("  y:", game["mixed_nash"][0])
    print("  z:", game["mixed_nash"][1])
    print("  Found mixed Nash:")
    print("  y:", list(y))
    print("  z:", list(z))
    print("Solver (method %d) took %s secs" % (method, duration))
    print("------")
    


def main():


    game_A = gameTemplate.copy()
    game_A["name"]       = "A"
    game_A["matrix"]     = np.array([[-1, -7, -3, -4], [-5, -6, -4, -5], [-7, -2, 0, -3]])
    game_A["pure_nash"]  = [(1, 2)]
    game_A["mixed_nash"] = [(0, 1, 0), (0, 0, 1, 0)]

    game_B = gameTemplate.copy()
    game_B["name"]       = "headsORtails"
    game_B["matrix"]     = np.array([[1, -1], [-1, 1]])
    game_B["pure_nash"]  = []
    game_B["mixed_nash"] = [(0.5, 0.5), (0.5, 0.5)]

    checkGame(game_A, method = 0)
    print("\n\n")
    checkGame(game_B, method = 0)

    return None


if __name__ == "__main__":
    main()
