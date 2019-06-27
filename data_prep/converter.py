from __future__ import division
import cv2
from osgeo import osr, gdal

class Converter:

    def __init__(self, conf):
        self.image = cv2.imread(conf['data_folder'] + conf['image_file'])
        self.file = gdal.Open(conf['data_folder'] + conf['image_file'])
        self.lng0 = float(conf['lng0'])
        self.lat0 = float(conf['lat0'])
        self.lng1 = float(conf['lng1'])
        self.lat1 = float(conf['lat1'])
        (self.height, self.width, c) = self.image.shape

    def _pix2coord_(self, x, y):
        lng = ((self.lng1- self.lng0) / self.width) * x + self.lng0
        lat = self.lat0 - ((self.lat0 - self.lat1) / self.height) * y

        return (lng, lat)

    def coord2pix(self, lng, lat):
        x = int((lng - self.lng0) * (self.width / (self.lng1 - self.lng0)))
        y = int((self.lat0 - lat) * (self.height / (self.lat0 - self.lat1)))
        return (x, y)

    def pix2coord(self, x, y):
        lng = ((self.lng1- self.lng0) / self.width) * x + self.lng0
        lat = self.lat0 - ((self.lat0 - self.lat1) / self.height) * y

        return (lng, lat)

    def _coord2pix_(self, lng, lat):
        a = abs(((self.x2 - self.x1) / (self.y2 - self.y1)) * (lat - self.y1) - self.x1)
        #print self.lng0, ' -> ', a
        b = abs(((self.x4 - self.x3) / (self.y4 - self.y3)) * (lat - self.y3) - self.x3)
        #print self.lng1, ' -> ', b
        c = abs(((self.y1 - self.y4) / (self.x4 - self.x1)) * (lng - self.x1) - self.y4)
        #print self.lat0, ' -> ', c
        d = abs(((self.y2 - self.y3) / (self.x3 - self.x2)) * (lng - self.x2) - self.y3)
        #print self.lat1, ' -> ', d

        x = int((lng - a) * (self.width / (b - a)))
        y = int((c - lat) * (self.height / (c - d)))
        return (x, y)

    def pixel(self, dx, dy):
        px = self.file.GetGeoTransform()[0]
        py = self.file.GetGeoTransform()[3]
        rx = self.file.GetGeoTransform()[1]
        ry = self.file.GetGeoTransform()[5]
        x = dx / rx + px
        y = dy / ry + py

        return x, y
