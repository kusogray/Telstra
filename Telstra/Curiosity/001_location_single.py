'''
Created on Feb 1, 2016
author: whmou

Feb 1, 2016     1.0.0     Init.

'''

from Telstra.util.CustomLogger import info as log
from Telstra.DataCollector.DataStratifier import stratifyData
from Telstra.Resources import Config
from Telstra.DataCollector.DataReader import DataReader as DataReader
from Telstra.DataAnalyzer.ModelFactory import ModelFactory as ModelFactory
from Telstra.util.CustomLogger import musicAlarm

if __name__ == '__main__':
    
    # 1. read in data
    expInfo = "location_only" + Config.osSep
    _basePath = Config.FolderBasePath + expInfo
    
    
    doTestFlag = False
    path = _basePath + "train.csv"
    testPath = _basePath + "test10.csv"
   
    # 1. read data
    dr = DataReader()
    dr.readInCSV( path, "train")
    if doTestFlag == True:
        dr.readInCSV(testPath , "test")
        
    # 2. stratify 60 % data and train location only
    newX, newY = stratifyData(dr._trainDataFrame, dr._ansDataFrame, 0.4)
    
    
    # 3. get all best model from newX
    fab = ModelFactory()
    fab._gridSearchFlag = True
    fab._n_iter_search = 1
    fab._expInfo = "001_location_only" 
    fab.getAllModels(newX, newY)
    
    # 4. test all data, output 3 ans as features
    
    
    musicAlarm()
    
    