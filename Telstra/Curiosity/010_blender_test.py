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
from Telstra.util.ModelUtils import getMatchNameModelPath

if __name__ == '__main__':
    
    
    # 1. read in data
    expNo = "010"
    expInfo = expNo + "_stack_each_feature" 
    _basePath = Config.FolderBasePath + expInfo + Config.osSep
    
        
#      4. test all data, output 3 ans as features
#     D:\Kaggle\Telstra\004_one_hot_resource_type\(Xgboost)_(2016-02-06_11_14_31).model
#     D:\Kaggle\Telstra\004_one_hot_resource_type\(Random_Forest)_(2016-02-06_11_24_09).model
#     D:\Kaggle\Telstra\004_one_hot_resource_type\(Extra_Trees)_(2016-02-06_11_30_52).model
#     D:\Kaggle\Telstra\004_one_hot_resource_type\(K_NN)_(2016-02-06_11_40_10).model

    modelFolder = _basePath + "models" + Config.osSep + "stacked" + Config.osSep
    
    clfNameList = []
    clfNameList.append("Extra_Trees")
    clfNameList.append("K_NN")
    clfNameList.append("Random_Forest")
    clfNameList.append("Xgboost")
    clfNameList.append("Logistic_Regression")
    
    testCsv = _basePath + "010_train_tobe.csv"
    dr = DataReader()
    newX, testY = dr.cvtPathListToDfList(testCsv, "train")
    
    for curModel in clfNameList:
        modelPath =  modelFolder + str(getMatchNameModelPath(modelFolder, curModel))
        tmpOutPath = _basePath + expNo + "_blender" + curModel + "_train.csv"
        tmpClf = loadModel( modelPath)
        log(tmpClf.predict_proba(newX))
        #outDf = pd.concat([newX, pd.DataFrame(tmpClf.predict_proba(newX))], axis=1)
        outDf = pd.DataFrame(tmpClf.predict_proba(newX))
        outDf.to_csv(tmpOutPath, sep=',', encoding='utf-8')
        #musicAlarm()

    