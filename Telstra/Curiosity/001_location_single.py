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
from Telstra.util.ModelUtils import loadModel
import pandas as pd

if __name__ == '__main__':
    
    # 1. read in data
    expInfo = "001_location_only" + Config.osSep
    _basePath = Config.FolderBasePath + expInfo
    
    
    doTestFlag = True
    path = _basePath + "001_train_tobe.csv"
    testPath = _basePath + "001_test_tobe.csv"
   
    # 1. read data
    dr = DataReader()
    dr.readInCSV( path, "train")
    newX, newY = dr._trainDataFrame, dr._ansDataFrame
    if doTestFlag == True:
        dr.readInCSV(testPath , "test")
        newX = dr._testDataFrame
        #newX = pd.DataFrame(newX[newX.columns[0]])
        print newX
    # 2. stratify 60 % data and train location only
#     newX, newY = stratifyData(dr._trainDataFrame, dr._ansDataFrame, 0.4)
    
    
    # 3. get all best model from newX
#     fab = ModelFactory()
#     fab._gridSearchFlag = True
#     fab._n_iter_search = 500
#     fab._expInfo = "001_location_only" 
#     fab.getAllModels(newX, newY)
    
    # 4. test all data, output 3 ans as features
    modelPath = _basePath+"(Xgboost)_(2016-02-03_18_39_14).model"
    tmpOutPath = _basePath + "001_submission_2.csv"
    tmpClf = loadModel( modelPath)
    log(tmpClf.predict_proba(newX))
    #outDf = pd.concat([newX, pd.DataFrame(tmpClf.predict_proba(newX))], axis=1)
    outDf = pd.DataFrame(tmpClf.predict_proba(newX))
    outDf.to_csv(tmpOutPath, sep=',', encoding='utf-8')
    musicAlarm()
    
    