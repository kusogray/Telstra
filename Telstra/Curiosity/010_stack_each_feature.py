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
from Telstra.util.ModelUtils import getMatchNameModelPath
import pandas as pd
from Telstra.Bartender.Blender import Blender


if __name__ == '__main__':
    
    
   # 1. read in data
    expNo = "010"
    expInfo = expNo + "_stack_each_feature" 
    _basePath = Config.FolderBasePath + expInfo + Config.osSep
    
#     
#     doTestFlag = True
#     tmpSingleOneHotFeatureName = ""
#     path = _basePath + expNo + "_train_" + tmpSingleOneHotFeatureName + "tobe.csv"
#     testPath = _basePath + expNo + "_test_tobe.csv"
#    
#     # 1. read data
#     dr = DataReader()
#     dr.readInCSV( path, "train")
#     newX, newY = dr._trainDataFrame, dr._ansDataFrame
#     if doTestFlag == True:
#         dr.readInCSV(testPath , "test")
#         newX = dr._testDataFrame
        #newX = pd.DataFrame(newX[newX.columns[0]])
        #print newX
        
    #newX, newY = stratifyData(dr._trainDataFrame, dr._ansDataFrame, 0.4)
    
    # 3. get all best model from newX
#     fab = ModelFactory()
#     fab._gridSearchFlag = True
#     fab._subFolderName = "stacked"
#     fab._n_iter_search = 250
#     fab._expInfo = expInfo
#     fab.getAllModels(newX, newY)
    
    # 4. test all data, output 3 ans as features
    #D:\Kaggle\Telstra\004_one_hot_resource_type\(Xgboost)_(2016-02-06_11_14_31).model
    #D:\Kaggle\Telstra\004_one_hot_resource_type\(Random_Forest)_(2016-02-06_11_24_09).model
    #D:\Kaggle\Telstra\004_one_hot_resource_type\(Extra_Trees)_(2016-02-06_11_30_52).model
    #D:\Kaggle\Telstra\004_one_hot_resource_type\(K_NN)_(2016-02-06_11_40_10).model
    #Logistic_Regression
    modelList = ["Xgboost","Random_Forest","Extra_Trees", "K_NN", "Logistic_Regression"]
    featureList = ["event_type", "log_feature", "resource_type", "severity_type"]
    
    for tmpFeature in featureList: 
        for tmpModel in modelList:
            subFolder = tmpFeature
            curModel = tmpModel
            
            tmpCsvPath = _basePath + expNo + "_" + tmpFeature +"_test_tobe.csv"
            dr = DataReader()
            dr.readInCSV(tmpCsvPath , "train")
            newX = dr._trainDataFrame
            modelFolder = _basePath + "models" + Config.osSep + subFolder + Config.osSep
            modelPath =  modelFolder + str(getMatchNameModelPath(modelFolder, curModel))
            tmpOutPath = _basePath + "010_" + curModel + "_stack_" + subFolder + "_test.csv"
            tmpClf = loadModel( modelPath)
            log(tmpClf.predict_proba(newX))
            outDf = pd.concat([newX, pd.DataFrame(tmpClf.predict_proba(newX))], axis=1)
            outDf = pd.DataFrame(tmpClf.predict_proba(newX))
            outDf.to_csv(tmpOutPath, sep=',', encoding='utf-8')
    #musicAlarm()
#     log("004 Done")