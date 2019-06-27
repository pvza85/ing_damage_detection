import pandas as pd
from converter import *

class Buildings:
    def __init__(self, conf):
        self.converter = Converter(conf)
        data = pd.read_csv(conf['data_folder'] + conf['building_file'])
        self.buildings = []
        for index in data['ORIG_FID'].unique():
            Lng = data['X'][data['ORIG_FID'] == index]
            Lat = data['Y'][data['ORIG_FID'] == index]
            self.buildings.append(Building(Lng, Lat, index + 1, self.converter, int(conf['window_size']) ))

class Building:
    def __init__(self, Lng, Lat, index, converter, window_size):
        self.lng0 = min(Lng)
        self.lat0 = max(Lat)
        self.lng1 = max(Lng)
        self.lat1 = min(Lat)
        self.x0, self.y0 = converter.coord2pix(self.lng0, self.lat0)
        self.x1, self.y1 = converter.coord2pix(self.lng1, self.lat1)
        self.index = index
        self.window_size = window_size

    def __str__(self):
        return ' lang0: {0} \n lat0: {1}\n x0: {2} \n y0: {3}\n lang1: {4} \n lat1: {5}\n x1: {6} \n y1: {7}\n'\
            .format(self.lng0, self.lat0, self.x0, self.y0, self.lng1, self.lat1, self.x1, self.y1 )
    def overlap(self, x0, y0, x1, y1):
        overlapX = float(min(self.f(x1-self.x0, self.window_size), self.f(self.x1 - x0, self.window_size))) / self.window_size
        #overlapY = float(min(self.f(self.y0 - y1, self.window_size), self.f(y0 - self.y1, self.window_size))) / self.window_size
        overlapY = float(min(self.f(y1 - self.y0, self.window_size), self.f(self.y1 - y0, self.window_size))) / self.window_size
        #if overlapX * overlapY > 0:
            #print overlapX, ' ', overlapY
        return overlapX * overlapY

    def f(self, n, size):
        res = 0
        if n < 0:
            res = 0
        elif n > size:
            res = size
        else:
            res = n
        return res