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
    expNo = "011"
    expInfo = expNo + "_remove_one_hot" 
    _basePath = Config.FolderBasePath + expInfo + Config.osSep
    
    path = _basePath + expNo + "_train_tobe.csv"
    testPath = _basePath + expNo + "_test_tobe.csv"
    
    dr = DataReader()
    dr.readInCSV( path, "train")
    newX, newY = dr._trainDataFrame, dr._ansDataFrame
    
    # Get all best model from newX
    fab = ModelFactory()
    fab._gridSearchFlag = True
#     fab._subFolderName = "stacked"
    fab._n_iter_search = 100
    fab._expInfo = expInfo
    fab.getAllModels(newX, newY)
    
    # Test all data
    modelList = ["Xgboost","Random_Forest","Extra_Trees", "K_NN", "Logistic_Regression"]
#     featureList = ["event_type", "log_feature", "resource_type", "severity_type"]
    
#     for tmpFeature in featureList:
    modelFolder = _basePath + "models" + Config.osSep 
    for tmpModel in modelList:  
        curModel = tmpModel
        
        dr = DataReader()
        newX = dr.cvtPathListToDfList(testPath, "test")
        
        modelPath =  modelFolder + str(getMatchNameModelPath(modelFolder, curModel))
        tmpOutPath = _basePath + "011_" + curModel + "_test_ans.csv"
        tmpClf = loadModel( modelPath)
        log(tmpClf.predict_proba(newX))
        outDf = pd.concat([newX, pd.DataFrame(tmpClf.predict_proba(newX))], axis=1)
        outDf = pd.DataFrame(tmpClf.predict_proba(newX))
        outDf.to_csv(tmpOutPath, sep=',', encoding='utf-8')
        
    
    #musicAlarm()
#     log("004 Done")