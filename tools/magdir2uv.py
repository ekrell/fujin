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



def main():

    options, args = parseOptions()
    if options == None:
        print("Bad options")
        return -1

    if options.reverse == True:
        print("Reverse option not yet supported.")
        return -1

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
                 (mag_raster.RasterCount, mag_raster.RasterCount))
            dimvalid = False
        if mag_shape[0] != dir_shape[0] or mag_shape[1] != dir_shape[1]:
            print("Magnitude: (%d, %d). directions: (%d, %d)" % \
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
            mag_shape[1], mag_shape[0], mag_raster.RasterCount, gdal.GDT_Byte)
        ucomponents_raster.SetGeoTransform(mag_raster.GetGeoTransform())
        ucomponents_raster.SetProjection(mag_raster.GetProjection())
        vcomponents_raster = gdal.GetDriverByName('GTiff').Create(options.vcomponent,
            mag_shape[1], mag_shape[0], mag_raster.RasterCount, gdal.GDT_Byte)
        vcomponents_raster.SetGeoTransform(mag_raster.GetGeoTransform())
        vcomponents_raster.SetProjection(mag_raster.GetProjection())
        # Calculate u, v
        for b in range(mag_raster.RasterCount):
            mag_array = mag_raster.GetRasterBand(b + 1).ReadAsArray()
            dir_array = dir_raster.GetRasterBand(b + 1).ReadAsArray()
            for row in range(mag_shape[0]):
                for col in range(mag_shape[1]):
                     u, v = magdir2uv(mag_array[row][col], dir_array[row][col])
                     ucomponents_array[b][row][col] = u
                     vcomponents_array[b][row][col] = v    
            ucomponents_raster.GetRasterBand(b + 1).WriteArray(ucomponents_array[b])            
            vcomponents_raster.GetRasterBand(b + 1).WriteArray(vcomponents_array[b])            
        # Write to disk
        ucomponents_raster.FlushCache()
        vcomponents_raster.FlushCache()
    
if __name__ == "__main__":
    main()

        
        





    

