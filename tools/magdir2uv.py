#!/usr/bin/python
from math import acos, cos, sin
import numpy as np
from osgeo import gdal
from math import cos, sin

def parseOptions():
    from optparse import OptionParser

    # Define options
    parser = OptionParser()
    parser.add_option("-m", "--magnitude",  dest = "magnitude",  metavar = "MAGNITUDE",
        help = "geotiff with vector magnitudes")
    parser.add_option("-d", "--direction",  dest = "direction",  metavar = "DIRECTION",
        help = "geotiff with vector directions")
    parser.add_option("-u", "--ucomponent", dest = "ucomponent", metavar = "UCOMPONENT",
        help = "geotiff for vector u components")
    parser.add_option("-v", "--vcomponent", dest = "vcomponent", metavar = "VCOMPONENT",
        help = "geotiff for vector v components")
    parser.add_option("-r", "--reverse",    dest = "reverse",    metavar = "REVERSE",
        action = "store_true", default = False,
        help = "reverse: becomes uv2magdir")
    parser.add_option("-s", "--scale",      dest = "scale",      metavar = "SCALE",
        default = 1.0, type = 'float',
        help = "scale magnitude")

    # Get options
    (options, args) = parser.parse_args()

    if   options.magnitude  is None \
      or options.direction  is None \
      or options.ucomponent is None \
      or options.vcomponent is None:

        return (None, None)
    else:
        return (options, args)

def magdir2uv(m, d):
    u = float(m) * cos(d)
    v = float(m) * sin(d)
    return (u, v)

def uv2magdir(u, v):
    vecA = (u, v)
    vecB = (1, 0)

    dotProd = sum( [vecA[i] * vecB[i] for i in range(len(vecA))] )
    magA = pow (sum( [vecA[i] * vecA[i] for i in range(len(vecA))] ), 0.5)
    magB = pow (sum( [vecB[i] * vecB[i] for i in range(len(vecB))] ), 0.5)

    if (magA * magB < 0.001):
         return (0.0, 1)
    cosTheta = dotProd / (magA * magB)
    if ( abs (cosTheta) >= 1):
        return (0.0, 1)

    magnitude = pow (vecA[0] * vecA[0] + vecA[1] * vecA[1],  0.5)
    direction = acos (cosTheta)
    return magnitude, direction

def main():

    options, args = parseOptions()
    if options == None:
        print("Bad options")
        return -1

    if options.reverse == True:
        # Load u, v rasters
        u_raster = gdal.Open(options.ucomponent)
        v_raster = gdal.Open(options.vcomponent)
        u_shape = u_raster.GetRasterBand(1).ReadAsArray().shape
        v_shape = v_raster.GetRasterBand(1).ReadAsArray().shape

        # Ensure same dimensions
        dimvalid = True
        if u_raster.RasterCount != v_raster.RasterCount:
            print("u-components: %d bands, v-components: %d bands" % \
                (u_raster.RasterCount, v_raster.RasterCount))
            dimvalid = False
        if u_shape[0] != v_shape[0] or u_shape[1] != v_shape[1]:
            print("u-components: (%d, %d), v-components: (%d, %d)" % \
                (u_shape[0], u_shape[1], v_shape[0], v_shape[1]))
            dimvalid = False
        if dimvalid == False:
            print("Rasters must have same dimensions")
            return -1

        # Initialize new mag, dir rasters
        mag_array = [np.zeros(u_shape) for b in range(u_raster.RasterCount)]
        dir_array = [np.zeros(u_shape) for b in range(u_raster.RasterCount)]
        # Init rasters
        mag_raster = gdal.GetDriverByName('GTiff').Create(options.magnitude,
            u_shape[1], u_shape[0], u_raster.RasterCount, gdal.GDT_Float32)
        mag_raster.SetGeoTransform(u_raster.GetGeoTransform())
        mag_raster.SetProjection(u_raster.GetProjection())
        dir_raster = gdal.GetDriverByName('GTiff').Create(options.direction,
            u_shape[1], u_shape[0], u_raster.RasterCount, gdal.GDT_Float32)
        dir_raster.SetGeoTransform(u_raster.GetGeoTransform())
        dir_raster.SetProjection(u_raster.GetProjection())
        # Calculate magnitude, direction
        for b in range(u_raster.RasterCount):
            u_array = u_raster.GetRasterBand(b + 1).ReadAsArray()
            v_array = v_raster.GetRasterBand(b + 1).ReadAsArray()
            for row in range(u_shape[0]):
                for col in range(u_shape[1]):

                     m, d = uv2magdir(u_array[row][col], v_array[row][col])

                     mag_array[b][row][col] = m
                     dir_array[b][row][col] = d
            # Scale magnitude
            mag_array[b] = mag_array[b] * options.scale
            mag_band = mag_raster.GetRasterBand(b + 1)
            mag_band.WriteArray(mag_array[b])
            dir_band = dir_raster.GetRasterBand(b + 1)
            dir_band.WriteArray(dir_array[b])
        # Write to disk
        mag_raster.FlushCache()
        dir_raster.FlushCache()
    else:
        # Load magnitude, direction rasters
        mag_raster = gdal.Open(options.magnitude)
        dir_raster = gdal.Open(options.direction)
        mag_shape  = mag_raster.GetRasterBand(1).ReadAsArray().shape
        dir_shape  = dir_raster.GetRasterBand(1).ReadAsArray().shape

        # Ensure same dimensions
        dimvalid = True
        if mag_raster.RasterCount != dir_raster.RasterCount:
            print("Magnitude: %d bands, directions: %d bands" % \
                 (mag_raster.RasterCount, dir_raster.RasterCount))
            dimvalid = False
        if mag_shape[0] != dir_shape[0] or mag_shape[1] != dir_shape[1]:
            print("Magnitude: (%d, %d), directions: (%d, %d)" % \
                 (mag_shape[0], mag_shape[1], dir_shape[0], dir_shape[1]))
            dimvalid = False
        if dimvalid == False:
            print("Rasters must have same dimensions")
            return -1

        # Initialize new u, v components
        ucomponents_array = [np.zeros(mag_shape) for b in range(mag_raster.RasterCount)]
        vcomponents_array = [np.zeros(mag_shape) for b in range(mag_raster.RasterCount)]
        # Init rasters
        ucomponents_raster = gdal.GetDriverByName('GTiff').Create(options.ucomponent,
            mag_shape[1], mag_shape[0], mag_raster.RasterCount, gdal.GDT_Float32)
        ucomponents_raster.SetGeoTransform(mag_raster.GetGeoTransform())
        ucomponents_raster.SetProjection(mag_raster.GetProjection())
        vcomponents_raster = gdal.GetDriverByName('GTiff').Create(options.vcomponent,
            mag_shape[1], mag_shape[0], mag_raster.RasterCount, gdal.GDT_Float32)
        vcomponents_raster.SetGeoTransform(mag_raster.GetGeoTransform())
        vcomponents_raster.SetProjection(mag_raster.GetProjection())
        # Calculate u, v
        for b in range(mag_raster.RasterCount):
            mag_array = mag_raster.GetRasterBand(b + 1).ReadAsArray()
            # Scale magnitude
            mag_array = mag_array * options.scale
            dir_array = dir_raster.GetRasterBand(b + 1).ReadAsArray()
            for row in range(mag_shape[0]):
                for col in range(mag_shape[1]):
                     u, v = magdir2uv(mag_array[row][col], dir_array[row][col])
                     ucomponents_array[b][row][col] = u
                     vcomponents_array[b][row][col] = v
            uband = ucomponents_raster.GetRasterBand(b + 1)
            uband.WriteArray(ucomponents_array[b])
            vband = vcomponents_raster.GetRasterBand(b + 1)
            vband.WriteArray(vcomponents_array[b])
        # Write to disk
        ucomponents_raster.FlushCache()
        vcomponents_raster.FlushCache()
if __name__ == "__main__":
    main()










