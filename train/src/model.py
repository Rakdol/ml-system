import tensorflow as tf


class HousingDataSet(object):
    def __init__(self, data_direcotry, transform):
        self.data_directory = data_direcotry
        self.transform = transform
        
