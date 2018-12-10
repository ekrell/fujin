import numpy as np
from mpl_toolkits.basemap import Basemap
from osgeo import gdal
import georaster
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors      import ListedColormap, BoundaryNorm
import matplotlib.patches as patches
from math import cos, sin

def plotPathStatHistory(history_df, plotsfile, dpi = 300):
    # history_df : dataframe of pathStat history
    plt.plot(history_df.index.values, history_df["work"], 'p')
    plt.ylabel("Work")
    plt.xlabel("Iteration")
    plt.title("Work Convergence")
    plt.savefig(plotsfile + "_work.eps", bbox_inches = "tight", format = "eps", dpi = dpi)
    plt.clf()

    plt.plot(history_df.index.values, history_df["waypoints"], 'p')
    plt.ylabel("Waypoints")
    plt.xlabel("Iteration")
    plt.title("Number of Waypoints Convergence")
    plt.savefig(plotsfile + "_waypoints.eps", bbox_inches = "tight", format = "eps", dpi = dpi)
    plt.clf()

    plt.plot(history_df.index.values, history_df["distance"], 'p')
    plt.ylabel("Distance")
    plt.xlabel("Iteration")
    plt.title("Distance Convergence")
    plt.savefig(plotsfile + "_distance.eps", bbox_inches = "tight", format = "eps", dpi = dpi)
    plt.clf()

    plt.plot(history_df.index.values, history_df["avg"], 'p')
    plt.ylabel("Average cost2go value")
    plt.xlabel("Iteration")
    plt.title("Average cost2go Convergence")
    plt.savefig(plotsfile + "_avg.eps", bbox_inches = "tight", format = "eps", dpi = dpi)
    plt.clf()
    
    plt.plot(history_df.index.values, history_df["avg_diff"], 'p')
    plt.ylabel("Average cost2go Change")
    plt.xlabel("Iteration")
    plt.title("Average cost2go Change Convergence")
    plt.savefig(plotsfile + "_avg_diff.eps", bbox_inches = "tight", format = "eps", dpi = dpi)
    plt.clf()



def plotRegion(occupancyRasterFile, occupancyGrid, plotsfile = None, width = 10):
    occ_raster = gdal.Open(occupancyRasterFile)
    cols = occ_raster.RasterXSize
    rows = occ_raster.RasterYSize
    height = ( float(rows) / float(cols) ) * float(width)
    f = plt.figure(figsize = (width, height))
    ax = f.add_subplot(111)
    region = georaster.SingleBandRaster(occ_raster, load_data=False)
    minx, maxx, miny, maxy = region.extent
    m = Basemap (projection = 'cyl',
        llcrnrlon = minx - .005,
        llcrnrlat = miny - .005,
        urcrnrlon = maxx + .005,
        urcrnrlat = maxy + .005,
        resolution = 'h')
    image = georaster.SingleBandRaster(occ_raster,
        load_data=(minx, maxx, miny, maxy), latlon=True)
    ax.imshow(image.r, extent=(0, cols, 0, rows), zorder=10, alpha=0.6)

    if plotsfile is not None:
        plt.savefig(plotsfile + "_region.eps", bbox_inches = "tight", format = "eps", dpi = 300)

    return ax

def plotPath(trace, waypoints, occupancyRasterFile, occupancyGrid, plotsfile = None, width = 10, init = True): 
    def plotit(trace, waypoints):
    
        ypoints = [occupancyGrid.shape[0] - w[0] for w in waypoints]
        xpoints = [w[1] for w in waypoints]
        plt.plot(xpoints, ypoints, 'x--')
        plt.scatter([trace[0][1]], [occupancyGrid.shape[0] - trace[0][0]],
                                 s = 250, marker = "8", color = "#ec73c9")
        plt.scatter([trace[len(trace) - 1][1]], [occupancyGrid.shape[0] - trace[len(trace) - 1][0]],
                                                          s = 250, marker = "8", color = "seagreen")
        return plt.gca()

    if init == True:
        plt.close("all")
        ax = plotRegion(occupancyRasterFile, occupancyGrid)
    plt.xlim(auto=False)
    plt.ylim(auto=False)
    ax = plotit(trace, waypoints)

    ax.set_rasterized(True)
    if plotsfile is not None:
        plt.savefig(plotsfile + "_path.eps", bbox_inches = "tight", format = "eps", dpi = 300)

    return ax

def plotVector(ugrids, vgrids, occupancyRasterFile, occupancyGrid, plotsfile = None, width = 10, init = True, sampleInterval = 10):

    plt.close("all")

    usamples_sum = None
    vsamples_sum = None

    for i in range(len(ugrids)):
        plt.clf()
        ax = plotRegion(occupancyRasterFile, occupancyGrid)
        plt.xlim(auto=False)
        plt.ylim(auto=False)

        ugrid = np.flipud(ugrids[i])
        vgrid = np.flipud(vgrids[i])

        rows = len(occupancyGrid)
        cols = len(occupancyGrid[0])

        ysamples = []
        xsamples = []
        usamples = []
        vsamples = []

        for row in range(0, rows - 1, sampleInterval):
            for col in range(0, cols - 1, sampleInterval):
                ysamples.append(row)
                xsamples.append(col)
                usamples.append(ugrid[row][col])
                vsamples.append(vgrid[row][col])

        if usamples_sum == None:
            usamples_sum = np.array(usamples)
        else:
            usamples_sum = usamples_sum + usamples
        if vsamples_sum == None:
            vsamples_sum = np.array(vsamples)
        else:
            vsamples_sum = vsamples_sum + vsamples

        ysamples = [rows - y for y in ysamples]
        plt.quiver(xsamples[::], ysamples[::], usamples[::], vsamples[::])
       
        ax.set_rasterized(True)
        if plotsfile is not None:
            plt.savefig(plotsfile + "_environment_" + str(i) + ".eps", bbox_inches = "tight", format = "eps", dpi = 300)

    plt.clf()
    ax = plotRegion(occupancyRasterFile, occupancyGrid)
    plt.xlim(auto=False)
    plt.ylim(auto=False)
    ysamples = [rows - y for y in ysamples]
    plt.quiver(xsamples[::], ysamples[::], usamples_sum[::], vsamples_sum[::])
     
    ax.set_rasterized(True)
    if plotsfile is not None:
        plt.savefig(plotsfile + "_environment" + ".eps", bbox_inches = "tight", format = "eps", dpi = 300)

    return ax



def plotActions(actiongrid, action2radians, occupancyRasterFile, occupancyGrid, plotsfile = None, 
                width = 10, init = True, sampleInterval = 10, magnitude = 1):

    def action2uv(action, magnitude, action2radians):
        if action == '*' or action == '-' or action == ' ':
            magnitude = 0
        u = magnitude * cos(action2radians[action])
        v = magnitude * sin(action2radians[action])
        return (u, v)

    rows = len(occupancyGrid)
    cols = len(occupancyGrid[0])

    #actiongrid = np.flipud(actiongrid)

    ysamples = []
    xsamples = []
    usamples = []
    vsamples = []

    for row in range(0, rows - 1, sampleInterval):
        for col in range(0, cols - 1, sampleInterval):
            ysamples.append(row)
            xsamples.append(col)
            u, v = action2uv(actiongrid[row][col], magnitude, action2radians)
            usamples.append(u)
            vsamples.append(v)

    
    plt.close("all")
    ax = plotRegion(occupancyRasterFile, occupancyGrid)
    plt.xlim(auto = False)
    plt.ylim(auto = False)

    ysamples = [rows - y for y in ysamples]
    plt.quiver(xsamples[::], ysamples[::], usamples[::], vsamples[::], scale=20, scale_units='inches')

    ax.set_rasterized(True)
    if plotsfile is not None:
        plt.savefig(plotsfile + "_actions" + ".png", bbox_inches = "tight", format = "png", dpi = 300)

    return ax

