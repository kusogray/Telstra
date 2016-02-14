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


if __name__ == '__main__':
    
    
   # 1. read in data
    expNo = "014"
    expInfo = expNo + "_one_hot_each_features" 
    _basePath = Config.FolderBasePath + expInfo + Config.osSep
    
    featureList = ["location", "event_type", "resource_type" , "severity_type", "log_feature"]
    
    ansPath = _basePath + "014_ans_array.csv"
    drAns = DataReader()
    drAns.readInCSV(ansPath, "train")
    newY = drAns._ansDataFrame
    
    
       
    for i in range(1,33):
        log( "start " + str(i) + "/32 ...")
        tmpCurFeatureList = []
        
        flagList =[]
        for i2 in range (0, 7- len(bin(i))):
            flagList.append(0)
        for i2 in range(2,len(bin(i))):
            flagList.append(int(bin(i)[i2]))
        
        for j in range(0,5):
            if flagList[j] ==1:
                tmpCurFeatureList.append(featureList[j])
        
        log(tmpCurFeatureList)        
        
        
        newX = pd.DataFrame()
        
        for tmpFeature in tmpCurFeatureList:
            path = _basePath + tmpFeature + "_train.csv"
            dr = DataReader()
            tmpX = dr.cvtPathListToDfList(path, "test")
            newX = pd.concat([newX, tmpX], axis=1)
        #log("feature len: " , len(newX))
            
        # Get all best model from newX
        fab = ModelFactory()
        fab._setXgboostTheradToOne = True
        fab._gridSearchFlag = True
        fab._subFolderName = "one_hot_each_" + str(i)
        fab._n_iter_search = 30
        fab._expInfo = expInfo
#         fab.getAllModels(newX, newY)
        fab.getRandomForestClf(newX, newY)
#         fab.getXgboostClf(newX, newY)
        log ( i , "/32 done..." )
    
    
    
   
    
    
    musicAlarm()
    # Test all data
    modelList = ["Xgboost","Random_Forest","Extra_Trees", "K_NN", "Logistic_Regression"]
#     featureList = ["event_type", "log_feature", "resource_type", "severity_type"]
    
#     for tmpFeature in featureList:
#     modelFolder = _basePath + "models" + Config.osSep 
#     for tmpModel in modelList:  
#         curModel = tmpModel
#         
#         dr = DataReader()
#         newX = dr.cvtPathListToDfList(testPath, "test")
#         
#         modelPath =  modelFolder + str(getMatchNameModelPath(modelFolder, curModel))
#         tmpOutPath = _basePath + "011_" + curModel + "_test_ans.csv"
#         tmpClf = loadModel( modelPath)
#         log(tmpClf.predict_proba(newX))
#         outDf = pd.concat([newX, pd.DataFrame(tmpClf.predict_proba(newX))], axis=1)
#         outDf = pd.DataFrame(tmpClf.predict_proba(newX))
#         outDf.to_csv(tmpOutPath, sep=',', encoding='utf-8')
        
    
    #musicAlarm()
#     log("004 Done")