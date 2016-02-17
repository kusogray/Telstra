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
from Telstra.util.ModelUtils import deleteModelFiles
import pandas as pd
from Telstra.Bartender.Blender import Blender
from test._mock_backport import inplace
import random


if __name__ == '__main__':
    
    
   # 1. read in data
    expNo = "016"
    expInfo = expNo + "_merge10_remove_id" 
    _basePath = Config.FolderBasePath + expInfo + Config.osSep
    
    featureList = ["location", "event_type", "resource_type" , "severity_type", "log_feature"]
    ans1List = []
    ans2List = []
#     ansPath = _basePath + "014_ans_array.csv"
#     drAns = DataReader()
#     drAns.readInCSV(ansPath, "train")
#     newY = drAns._ansDataFrame

    tmpPath = _basePath + "016_train_tobe.csv"
    dr = DataReader()
    dr.readInCSV(tmpPath, "train")
    newX = dr._trainDataFrame
    newY = dr._ansDataFrame


    
    # Get all best model from newX
    fab = ModelFactory()
    #fab._setXgboostTheradToOne = True
    fab._gridSearchFlag = True
    fab._singleModelMail = True
    fab._subFolderName = "removeId"  
    fab._n_iter_search = 1
    fab._expInfo = expInfo
# #         fab.getAllModels(newX, newY)
    #fab.getRandomForestClf(newX, newY)
    fab.getXgboostClf(newX, newY)
#         fab.getXgboostClf(newX, newY)
#    log ( i , "/32 done..." )
    
    
    
   
    
    
    musicAlarm()
    # Test all data
    modelList = ["Xgboost","Random_Forest","Extra_Trees", "K_NN", "Logistic_Regression"]
#     featureList = ["event_type", "log_feature", "resource_type", "severity_type"]
    
#     for tmpFeature in featureList:
    modelFolder = _basePath + "models" + Config.osSep + "location_log_feature_over_sampling" + Config.osSep
#     for tmpModel in modelList:  
#         curModel = tmpModel
#          
#     testPath = _basePath + "location_log_feature_test3.csv"
#     dr = DataReader()
#     newX = dr.cvtPathListToDfList(testPath, "test")
#     curModel = "Random_Forest"
#        
#     modelPath =  modelFolder + str(getMatchNameModelPath(modelFolder, curModel))
#     tmpOutPath = _basePath + expNo +"_" + curModel + "_test_ans.csv"
#     tmpClf = loadModel( modelPath)
#     log(tmpClf.predict_proba(newX))
#     outDf = pd.concat([newX, pd.DataFrame(tmpClf.predict_proba(newX))], axis=1)
#     outDf = pd.DataFrame(tmpClf.predict_proba(newX))
#     outDf.to_csv(tmpOutPath, sep=',', encoding='utf-8')
#         
    
    #musicAlarm()
#     log("004 Done")