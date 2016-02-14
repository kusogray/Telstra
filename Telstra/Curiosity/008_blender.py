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
from Telstra.Bartender.Blender import Blender

if __name__ == '__main__':
    
    
    # 1. read in data
    expNo = "008"
    expInfo = expNo + "_blender" 
    _basePath = Config.FolderBasePath + expInfo + Config.osSep
    
    doTestFlag = False
    path = _basePath + expNo + "_train_tobe.csv"
    testPath = _basePath + expNo + "_test_tobe.csv"
    outputPath = _basePath + expNo + "blender_train.csv"
    
    # 1. read data
    dr = DataReader()
    tmpDfList = []
    tmpPath = _basePath + "008_submission_1_train_Extra_Trees.csv"
    newX, newY =  dr.cvtPathListToDfList(tmpPath, "train")
    tmpDfList.append(newX)
    
    tmpPath = _basePath + "008_submission_1_train_K_NN.csv"
    newX, newY =  dr.cvtPathListToDfList(tmpPath, "train")
    tmpDfList.append(newX)
    
    tmpPath = _basePath + "008_submission_1_train_Random_Forest.csv"
    newX, newY =  dr.cvtPathListToDfList(tmpPath, "train")
    tmpDfList.append(newX)
    
    tmpPath = _basePath + "008_submission_1_train_Xgboost.csv"
    newX, newY =  dr.cvtPathListToDfList(tmpPath, "train")
    tmpDfList.append(newX)
    
    clfNameList = []
    clfNameList.append("Extra_Trees")
    clfNameList.append("K_NN")
    clfNameList.append("RandomForest")
    clfNameList.append("Xgboost")
    
    
    b1 = Blender(clfNameList, tmpDfList, newY)
    b1.autoFlow(1000, outputPath)
    
    if doTestFlag == True:
        dr.readInCSV(testPath , "test")
        newX = dr._testDataFrame
        #newX = pd.DataFrame(newX[newX.columns[0]])
        #print newX
 
    
    # 3. get all best model from newX
#     fab = ModelFactory()
#     fab._gridSearchFlag = True
#     fab._n_iter_search = 100
#     fab._expInfo = expInfo
#     fab.getXgboostClf(newX, newY)
    
    # 4. test all data, output 3 ans as features
    #D:\Kaggle\Telstra\004_one_hot_resource_type\(Xgboost)_(2016-02-06_11_14_31).model
    #D:\Kaggle\Telstra\004_one_hot_resource_type\(Random_Forest)_(2016-02-06_11_24_09).model
    #D:\Kaggle\Telstra\004_one_hot_resource_type\(Extra_Trees)_(2016-02-06_11_30_52).model
    #D:\Kaggle\Telstra\004_one_hot_resource_type\(K_NN)_(2016-02-06_11_40_10).model

#     modelPath = _basePath+"(K_NN)_(2016-02-06_11_40_10).model"
#     tmpOutPath = _basePath + "004_submission_1_K_NN.csv"
#     tmpClf = loadModel( modelPath)
#     log(tmpClf.predict_proba(newX))
#     outDf = pd.concat([newX, pd.DataFrame(tmpClf.predict_proba(newX))], axis=1)
#     outDf = pd.DataFrame(tmpClf.predict_proba(newX))
#     outDf.to_csv(tmpOutPath, sep=',', encoding='utf-8')
#     musicAlarm()
    
    