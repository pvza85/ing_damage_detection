import sys
sys.path.append('../')
from osgeo import osr, gdal
import imutils

def if_geotiff(file_name):
    '''
    read a tiff file and return if it is geotiff or not
    :param file_name: string
    :return: Boolean
    '''
    res = False
    ds = gdal.Open(file_name)
    try:
        ds = gdal.Open(file_name)
        gt = ds.GetGeoTransform()
        if gt[0] != 0.0:
            res = True
    except:
        pass

    return res

def get_dimensions(file_name):
    '''
    get a geotiff file name and return its dimensions
    :param file_name: geotiff file name (string)
    :return: width, heigt, channel
    '''
    ds = gdal.Open(file_name)
    width = ds.RasterXSize
    height = ds.RasterYSize
    channel = ds.RasterCount
    return width, height, channel

def get_corners(file_name):
    '''
    get a file name and return coordination of corners
    :param file_name: geotiff file name (string)
    :return: minx, miny, maxx, maxy
    '''
    width, height, _ = get_dimensions(file_name)

    ds = gdal.Open(file_name)
    gt = ds.GetGeoTransform()

    '''
    adfGeoTransform[0] /* top left x */
    adfGeoTransform[1] /* w-e pixel resolution */
    adfGeoTransform[2] /* rotation, 0 if image is "north up" */
    adfGeoTransform[3] /* top left y */
    adfGeoTransform[4] /* rotation, 0 if image is "north up" */
    adfGeoTransform[5] /* n-s pixel resolution */
    '''
    minx = gt[0]
    miny = gt[3] + width * gt[4] + height * gt[5]
    maxx = gt[0] + width * gt[1] + height * gt[2]
    maxy = gt[3]
    return minx, miny, maxx, maxy

def convert_to_wgs84(file_name, (x, y)):
    '''
    get geotiff name and a point (x,y) in pixcels
    :param file_name: string
    :return: longitude and latitude of (x, y)
    '''
    # get the existing coordinate system
    ds = gdal.Open(file_name)
    old_cs = osr.SpatialReference()
    old_cs.ImportFromWkt(ds.GetProjectionRef())

    # create the new coordinate system
    wgs84_wkt = """
    GEOGCS["WGS 84",
        DATUM["WGS_1984",
            SPHEROID["WGS 84",6378137,298.257223563,
                AUTHORITY["EPSG","7030"]],
            AUTHORITY["EPSG","6326"]],
        PRIMEM["Greenwich",0,
            AUTHORITY["EPSG","8901"]],
        UNIT["degree",0.01745329251994328,
            AUTHORITY["EPSG","9122"]],
        AUTHORITY["EPSG","4326"]]"""
    new_cs = osr.SpatialReference()
    new_cs.ImportFromWkt(wgs84_wkt)

    # create a transform object to convert between coordinate systems
    transform = osr.CoordinateTransformation(old_cs, new_cs)
    latlong = transform.TransformPoint((x, y))
    return latlong[0], latlong[1]

def convert_from_wgs84(file_name, (x, y)):
    '''
    get geotiff name and a point (x,y) in pixcels
    :param file_name: string
    :return: longitude and latitude of (x, y)
    '''
    # get the existing coordinate system
    ds = gdal.Open(file_name)
    old_cs = osr.SpatialReference()
    old_cs.ImportFromWkt(ds.GetProjectionRef())

    # create the new coordinate system
    wgs84_wkt = """
    GEOGCS["WGS 84",
        DATUM["WGS_1984",
            SPHEROID["WGS 84",6378137,298.257223563,
                AUTHORITY["EPSG","7030"]],
            AUTHORITY["EPSG","6326"]],
        PRIMEM["Greenwich",0,
            AUTHORITY["EPSG","8901"]],
        UNIT["degree",0.01745329251994328,
            AUTHORITY["EPSG","9122"]],
        AUTHORITY["EPSG","4326"]]"""
    new_cs = osr.SpatialReference()
    new_cs.ImportFromWkt(wgs84_wkt)

    # create a transform object to convert between coordinate systems
    transform = osr.CoordinateTransformation( new_cs, old_cs)
    latlong = transform.TransformPoint(x, y)
    return latlong[0], latlong[1]

def pyramid(image, scale=1.5, minSize=(30, 30)):
    # yield the original image
    yield image

    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)

        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break

        # yield the next image in the pyramid
        yield image

def sliding_window(image, stepSize, windowSize, progress=(0,0)):
    # slide a window across the image
    startPoint = progress[0]
    for y in xrange(progress[1], image.shape[0], stepSize):
        for x in xrange(startPoint, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
        startPoint = 0