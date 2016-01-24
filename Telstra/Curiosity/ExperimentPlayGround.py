'''
Created on Jan 24, 2016

@author: whmou
'''

from Telstra.DataCollector.DataReader import DataReader as DataReader
from Telstra.DataAnalyzer.ModelFactory import ModelFactory as ModelFactory
import os

def exp():
    _basePath =""
    if os.name == 'nt':
        _basePath = "D:\\Kaggle\\Telstra\\"
    else:
        _basePath = "/Users/whmou/Kaggle/Telstra/"
        
    path = _basePath + "train_merge7.csv"
    testPath = _basePath + "test7.csv"
    # 1. read data
    dr = DataReader()
    dr.readInCSV( path, "train")
    dr.readInCSV(testPath , "test")
    
    # 2. run models
    #print dr._trainDataFrame.as_matrix
    fab = ModelFactory()
    rfClf = fab.getRandomForestClf(dr._trainDataFrame, dr._ansDataFrame)
    print rfClf.predict_proba(dr._testDataFrame)
    return rfClf.predict_proba(dr._testDataFrame)

if __name__ == '__main__':
    exp()
    