'''
Created on Jan 24, 2016

@author: whmou
'''

import pandas as pd


class DataReader(object):
    '''
    classdocs
    '''
    _trainDataFrame, _testDataFream, _ansDataFrame = [],[],[]

    def __init__(self, ):
        '''
        Constructor
        '''
    def readInCSV(self, path, mode):
        # # 1. read csv data in
        df = pd.read_csv(path, header=0, sep=',')
        if mode.lower() == "train":
            self._ansDataFrame = df[df.columns[0]]
            self._trainDataFrame = df[df.columns[1:]]
        else:
            self._testDataFrame = df
    
