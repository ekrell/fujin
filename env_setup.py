from math import acos, cos, sin, ceil, pi
import numpy as np
import os.path
from PIL import Image
from os.path import splitext
from osgeo import gdal


def parseOptions():
    from optparse import OptionParser

    # Setup settings dictionary
    settings = { "iterations"  : 1,
                 "occupancy"   : None,
                 "ucomponents" : None,
                 "vcomponents" : None,
                 "weights"     : None,
                 "weightgrids" : None,
                 "errors"      : None,
                 "errorgrids"  : None,
                 "start"       : None,
                 "target"      : None,
                 "files"       : { "cost2go"    : None,
                                   "work2go"    : None,
                                   "actiongrid" : None,
                                   "pickle"     : None,
                                   "pandas"     : None,
                                   "plots"      : None,
                                 },
                 "verbose"     : False,
                 "reuse"       : False,
                 "bounds"      : { "upperleft"  : None,
                                   "lowerright" : None,
                                 },
               }

    # Define options
    parser = OptionParser()
    parser.add_option("-i", "--iterations",    dest = "iterations",   metavar = "ITERATIONS",
            default = 1,
            help = "number of solver iterations")
    parser.add_option("-o", "--occupancy",     dest = "occupancy",    metavar = "OCCUPANCY",
            help = "list of occupancy images (csv)")
    parser.add_option("-u", "--ucomponents",   dest = "ucomponents",  metavar = "UCOMPONENTS",
            help = "list of u vector component images (csv)")
    parser.add_option("-v", "--vcomponents",   dest = "vcomponents",  metavar = "VCOMPONENTS",
            help = "list of v vector component images (csv)")
    parser.add_option("-w", "--weights",       dest = "weights",      metavar = "WEIGHTS",
            help = "list of vector weights (csv)")
    parser.add_option("--weightgrids",         dest = "weightgrids",  metavar = "WEIGHTGRIDS",
            help = "list of vector weight grids (csv)")
    parser.add_option("-e", "--errors",        dest = "errors",       metavar = "ERRORS",
            help = "list of vector errors (csv)")
    parser.add_option("--errorgrids",          dest = "errorgrids",   metavar = "ERRORGRIDS",
            help = "list of vector error grids (csv)")
    parser.add_option("-s", "--start",         dest = "start",        metavar = "START",
            help = "start position as row,col")
    parser.add_option("-t", "--target",        dest = "target",       metavar = "TARGET",
            help = "target position as row,col")
    parser.add_option("-c", "--costfile",      dest = "costfile",     metavar = "COSTFILE",
            help = "file to store grid with traveler action costs for each cell")
    parser.add_option("-x", "--workfile",      dest = "workfile",     metavar = "WORKFILE",
            help = "file to store grid with traveler applied work for each cell")
    parser.add_option("-a", "--actionfile",    dest = "actionfile",   metavar = "ACTIONFILE",
            help = "file to store grid with traveler actions for each cell")
    parser.add_option("--picklefile",          dest = "picklefile",   metavar = "PICKLEFILE",
            help = "file to store convergence history as pickle")
    parser.add_option("--pandasfile",          dest = "pandasfile",   metavar = "PANDASFILE",
            help = "file to store convergence history as pandas")
    parser.add_option("--plotsfiles",          dest = "plotsfile",    metavar = "PLOTSFILE",
            help = "file (prefix) to store convergence history plots. Leave off extention")
    parser.add_option("--verbose",             dest = "verbose",      metavar = "VERBOSE",
            action = "store_true", default = False,
            help = "Display messages during execution")
    parser.add_option("-r", "--reuse",         dest = "reuse",        metavar = "REUSE",
            action = "store_true", default = False,
            help = "Reuse existing actiongrid for goto planning")
    parser.add_option("-b", "--bounds",        dest = "bounds",       metavar = "BOUNDS",
            help = "Raster boundaries as x1,y1,x2,y2")

    # Get options
    (options, args) = parser.parse_args()

    # Check that required arguments exist
    if     options.occupancy  is None \
        or options.start      is None \
        or options.target     is None \
        or options.costfile   is None \
        or options.workfile   is None \
        or options.actionfile is None:

        (options, args, settings) = None, None, None
        return (options, args, settings)


    settings["files"]["cost2go"]    = options.costfile
    settings["files"]["work2go"]    = options.workfile
    settings["files"]["actiongrid"] = options.actionfile
    settings["files"]["pickle"]     = options.picklefile
    settings["files"]["pandas"]     = options.pandasfile
    settings["files"]["plots"]      = options.plotsfile
    settings["iterations"]          = int(options.iterations)

    settings["occupancy"] = options.occupancy.split(",")

    if options.ucomponents is not None:
        settings["ucomponents"] = options.ucomponents.split(",")
    if options.vcomponents is not None:
        settings["vcomponents"] = options.vcomponents.split(",")
    if options.weights     is not None:
        settings["weights"]     = [float(w) for w in options.weights    .split(",")]
    if options.errors is not None:
        settings["errors"]      = [float(e) for e in options.errors     .split(",")]
    if options.weightgrids is not None:
        settings["weightgrids"] = options.weightgrids.split(",")
    if options.errorgrids is not None:
        settings["errorgrids"]  = options.errorgrids.split(",")

    try:
        settings["start"]  = (int(options.start.split(",")[0]),
                              int(options.start.split(",")[1]))
        settings["target"] = (int(options.target.split(",")[0]),
                              int(options.target.split(",")[1]))
    except:
        (options, args, settings) = None, None, None
        return (options, args, settings)


    # Esnure that all lists related to the vectors are of same length
    if settings["ucomponents"] is not None or settings["vcomponents"] is not None:
       try:
            if len(settings["ucomponents"]) != len(settings["vcomponents"]):
                (options, args, settings) = None, None, None
                return (options, args, settings)
       except:
            (options, args, settings) = None, None, None
            return (options, args, settings)


    # Ensure all files are truly files
    fileCheck = True # Initially assume true, until otherwise seen
    for o in settings["occupancy"]:
        if os.path.isfile(o) == False:
            fileCheck = False
    if settings["ucomponents"] is not None or settings["vcomponents"] is not None:
        for u in settings["ucomponents"]:
            if os.path.isfile(u) == False:
                fileCheck = False
        for v in settings["vcomponents"]:
            if os.path.isfile(v) == False:
                fileCheck = False
    if fileCheck == False:
        (options, args, settings) = None, None, None
        return (options, args, settings)

    # Misc options
    settings["verbose"] = options.verbose
    settings["reuse"]   = options.reuse


    # Boundaries
    if options.bounds is not None:
        bounds = [int(s) for s in options.bounds.split(",")]
        settings["bounds"]["upperleft"]  = (bounds[0], bounds[1])
        settings["bounds"]["lowerright"] = (bounds[2], bounds[3])
        if settings["start"][0]  <  settings["bounds"]["upperleft"][0]  or \
           settings["start"][0]  >= settings["bounds"]["lowerright"][0] or \
           settings["start"][1]  <  settings["bounds"]["upperleft"][1]  or \
           settings["start"][1]  >= settings["bounds"]["lowerright"][1] or \
           settings["target"][0] <  settings["bounds"]["upperleft"][0]  or \
           settings["target"][0] >= settings["bounds"]["lowerright"][0] or \
           settings["target"][1] <  settings["bounds"]["upperleft"][1]  or \
           settings["target"][1] >= settings["bounds"]["lowerright"][1]:
            (options, args, settings) = None, None, None
            return (options, args, settings)





    return (options, args, settings)

def getTraveler(start, target, speed_cps, travelType):

    travelTypes = ["4way", "8way"]

    action2radians = { ">"  : 0.0,
                       "UR" : pi / 4.0,
                       "^"  : pi / 2.0,
                       "UL" : pi * 0.75,
                       "<"  : pi,
                       "DL" : pi * 1.25,
                       "v"  : pi * 1.5,
                       "DR" : pi * 1.75,
                       "*"  : 0,
                       "-"  : 0,
                       " "  : 0,
                    }

    traveler = { "start"          : start,
                 "target"         : target,
                 "actionspace"    : None,
                 "action2radians" : action2radians,
                 "speed_cps"      : speed_cps,
               }

    if   travelType == travelTypes[0]:
        traveler["actionspace"] = ["*", "^",  "v",  "<",  ">"]
    elif travelType == travelTypes[1]:
        traveler["actionspace"] = ["*", "^",  "v",  "<",  ">",
                                        "UL", "UR", "DL", "DR"]
    return traveler



def printEnv(env):
    print("------")
    print("Start coordinates")
    print("    {}".format(env["start"]))
    print("Target coordinates")
    print("    {}".format(env["target"]))
    print("Region images:")
    for i in range(len(env["occupancy"])):
        print("    {}".format(env["occupancy"][i]))

    print("Vector u images:")
    if env["ucomponents"] is not None:
        for i in range(len(env["ucomponents"])):
            print("    Vector {} : {}".format(i, env["ucomponents"][i]))
    else:
        print("none")
    print("Vector v images:")
    if env["vcomponents"] is not None:
        for i in range(len(env["vcomponents"])):
            print("    Vector {} : {}".format(i, env["vcomponents"][i]))
    else:
        print("none")
    #print("Vector weights:")
    #for i in range(len(env["weights"])):
    #    print("    Vector {} : {}".format(i, env["weights"][i]))
    #print("Vector errors:")
    #for i in range(len(env["errors"])):
    #    print("    Vector {} : +/- {}".format(i, env["errors"][i]))
    print("------")



def getOccupancyGrid(occupancyImageFiles):

    # Creates single 2D numpy array using all input files
    # such that if any file indicates an occupied cell,
    # the cell is occupied in the result array


    # occupancyImageFiles:
    # Example:
    #     [ data/EXP1_region_1.png, data/EXP1_region_1.png ]

    occgrid = None
    name, ext = splitext(occupancyImageFiles[0])

    if ext == ".png":
        grids = [np.asarray(Image.open(f)) for f in occupancyImageFiles]
        occgrid = np.zeros((grids[0].shape[0], grids[0].shape[1]))
        for row in range(len(occgrid)):
            for col in range(len(occgrid[0])):
                for g in range(len(grids)):
                    for c in range(4):
                        if (grids[g][row][col][c] != 255):
                            occgrid[row][col] = 1

    if ext == ".tif":
        r = gdal.Open(occupancyImageFiles[0])
        occgrid = r.GetRasterBand(1).ReadAsArray()
        occgrid = 0 * occgrid
        for f in occupancyImageFiles:
            r = gdal.Open(f)
            for band in range(r.RasterCount):
                occgrid = occgrid + r.GetRasterBand(band + 1).ReadAsArray()

    return occgrid


def getComponentGrid(componentImageFile, band = 1):

    compgrid = None
    name, ext = splitext(componentImageFile)

    if ext == ".png":
        # Creates 2D numpy array where the grayscale
        # value of the input image files determined
        # the proportion of the input maxval at each cell
        imgarray = np.array(Image.open(componentImageFile).convert('LA'))
        compgrid = np.zeros((imgarray.shape[0], imgarray.shape[1]))
        for row in range(len(compgrid)):
            for col in range(len(compgrid[0])):
                compgrid[row][col] = imgarray[row][col][0] / 255.0

    if ext == ".tif":
        comp = gdal.Open(componentImageFile)
        compgrid = comp.GetRasterBand(1).ReadAsArray()

    return compgrid


def getVectorGrids(ucomponentImageFiles, vcomponentImageFiles, occgrid):

    if ucomponentImageFiles is None or \
       vcomponentImageFiles is None:
        ugrids = [None]
        ugrids[0] = np.zeros(occgrid.shape)
        vgrids = [None]
        vgrids[0] = np.zeros(occgrid.shape)
    else:
        numGrids = len(ucomponentImageFiles)

        ugrids = [None for u in range(numGrids)]
        vgrids = [None for v in range(numGrids)]

        for i in range(numGrids):
            ugrids[i] = getComponentGrid(ucomponentImageFiles[i])
            vgrids[i] = getComponentGrid(ucomponentImageFiles[i])
    return ugrids, vgrids




def getWeightGrids(occgrid, weights, weightgridFiles, numGrids):

    weightgrids = [None for w in range(numGrids)]

    # Priority given to weightgrid files
    if weightgridFiles is not None:
        for i in range(numGrids):
            weightgrids[i] = np.loadtxt(weightgridFiles[i])
        return weightgrids

    # Then to a single value: assigned to all cells
    if weights is not None:
        for i in range(numGrids):
            weightgrids[i] = np.ones(occgrid.shape) * weights[i]
        return weightgrids

    # Else weight is one
    for i in range(numGrids):
        weightgrids[i] = np.ones(occgrid.shape)

    return weightgrids


def getErrorGrids(occgrid, errors, errorgridFiles, numGrids):

    errorgrids = [None for w in range(numGrids)]

    # Priority given to errorgrid files
    if errorgridFiles is not None:
        for i in range(numGrids):
            errorgrids[i] = np.loadtxt(errorgridFiles[i])
        return errorgrids

    # Then to a single value: assigned to all cells
    if errors is not None:
        for i in range(numGrids):
            errorgrids[i] = np.ones(occgrid.shape) * errors[i]
        return errorgrids

    # Else error is one
    for i in range(numGrids):
        errorgrids[i] = np.zeros(occgrid.shape)
    return errorgrids


