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
    expNo = "015"
    expInfo = expNo + "_over_sampling" 
    _basePath = Config.FolderBasePath + expInfo + Config.osSep
    
    featureList = ["location", "event_type", "resource_type" , "severity_type", "log_feature"]
    ans1List = []
    ans2List = []
#     ansPath = _basePath + "014_ans_array.csv"
#     drAns = DataReader()
#     drAns.readInCSV(ansPath, "train")
#     newY = drAns._ansDataFrame

    tmpPath = _basePath + "location_log_feature_train.csv"
    dr = DataReader()
    dr.readInCSV(tmpPath, "test")
    newX = dr._trainDataFrame
    newY = dr._ansDataFrame
    newX = pd.concat([newY, newX], axis =1)
#     
#     logFeaturePath = _basePath + "log_feature_train.csv"
#     dr = DataReader()
#     dr.readInCSV(logFeaturePath, "test")
#     newX = dr._testDataFrame
    
#     for i, tmpVal in enumerate (newY):
#         if tmpVal == 1:
#             ans1List.append(i)
#         elif tmpVal ==2:
#             ans2List.append(i)
     
#     for i in range(0,2913):
# #     for i in range(0,3):
#         tmpIdx = ans1List[random.randint(0, len(ans1List)-1)]
#         tmpAnsCol = pd.DataFrame()
#         #tmpRow = pd.concat([tmpAnsCol, pd.DataFrame(newX.iloc()[tmpIdx])], axis =1)
#         newX = newX.append(newX.iloc()[tmpIdx])
#         newX[newX.columns[0]][len(newX)-1] = 1
#         print i
#      
#     for i in range(0,4058):
# #     for i in range(0,3):
#         tmpIdx = ans2List[random.randint(0, len(ans2List)-1)]
#         tmpAnsCol = pd.DataFrame()
#         tmpAnsCol[0] = 2
#         newX = newX.append(newX.iloc()[tmpIdx])
#         newX[newX.columns[0]][len(newX)-1] = 2
#         print i
#      
#     #print newX.iloc()[0]
#     tmpOutPath = _basePath + "location_log_feature_over_sampling.csv"
#     print len(newX)
#     newX.to_csv(tmpOutPath, sep=',', encoding='utf-8')
    #
    #print len(newX)        
    # Get all best model from newX
#     fab = ModelFactory()
#     fab._setXgboostTheradToOne = True
#     fab._gridSearchFlag = True
#     fab._subFolderName = "location_log_feature_over_sampling"  
#     fab._n_iter_search = 30
#     fab._expInfo = expInfo
# #         fab.getAllModels(newX, newY)
#     fab.getRandomForestClf(newX, newY)
#     fab.getXgboostClf(newX, newY)
#         fab.getXgboostClf(newX, newY)
#    log ( i , "/32 done..." )
    
    
    
   
    
    
   # musicAlarm()
    # Test all data
    modelList = ["Xgboost","Random_Forest","Extra_Trees", "K_NN", "Logistic_Regression"]
#     featureList = ["event_type", "log_feature", "resource_type", "severity_type"]
    
#     for tmpFeature in featureList:
    modelFolder = _basePath + "models" + Config.osSep + "location_log_feature_over_sampling" + Config.osSep
#     for tmpModel in modelList:  
#         curModel = tmpModel
#          
#     testPath = _basePath + "location_log_feature_test2.csv"
#     dr = DataReader()
#     newX = dr.cvtPathListToDfList(testPath, "test")
#     curModel = "Xgboost"
#       
#     modelPath =  modelFolder + str(getMatchNameModelPath(modelFolder, curModel))
#     tmpOutPath = _basePath + expNo +"_" + curModel + "_test_ans.csv"
#     tmpClf = loadModel( modelPath)
#     log(tmpClf.predict_proba(newX))
#     outDf = pd.concat([newX, pd.DataFrame(tmpClf.predict_proba(newX))], axis=1)
#     outDf = pd.DataFrame(tmpClf.predict_proba(newX))
#     outDf.to_csv(tmpOutPath, sep=',', encoding='utf-8')
        
    
    #musicAlarm()
#     log("004 Done")