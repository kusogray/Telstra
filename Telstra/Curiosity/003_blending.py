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
    expInfo = "003_one_hot_" + Config.osSep
    _basePath = Config.FolderBasePath + expInfo
    
    
    doTestFlag = True
    path = _basePath + "002_train_tobe.csv"
    testPath = _basePath + "002_test_tobe.csv"
   
    # 1. read data
    dr = DataReader()
    dr.readInCSV( path, "train")
    newX, newY = dr._trainDataFrame, dr._ansDataFrame
    if doTestFlag == True:
        dr.readInCSV(testPath , "test")
        newX = dr._testDataFrame
        #newX = pd.DataFrame(newX[newX.columns[0]])
        #print newX
 
    
    # 3. get all best model from newX
#     fab = ModelFactory()
#     fab._gridSearchFlag = True
#     fab._n_iter_search = 30
#     fab._expInfo = "002_blending" 
#     fab.getAllModels(newX, newY)
    
    # 4. test all data, output 3 ans as features
    #D:\Kaggle\Telstra\002_blending\(Xgboost)_(2016-02-03_20_09_03).model
    #D:\Kaggle\Telstra\002_blending\(Random_Forest)_(2016-02-03_20_16_16).model
    #D:\Kaggle\Telstra\002_blending\(Extra_Trees)_(2016-02-03_20_21_58).model
    #D:\Kaggle\Telstra\002_blending\(K_NN)_(2016-02-03_20_32_22).model
    
    modelPath = _basePath+"(K_NN)_(2016-02-03_20_32_22).model"
    tmpOutPath = _basePath + "002_submission_1_K_NN.csv"
    tmpClf = loadModel( modelPath)
    log(tmpClf.predict_proba(newX))
    outDf = pd.concat([newX, pd.DataFrame(tmpClf.predict_proba(newX))], axis=1)
    outDf = pd.DataFrame(tmpClf.predict_proba(newX))
    outDf.to_csv(tmpOutPath, sep=',', encoding='utf-8')
    musicAlarm()
    
    