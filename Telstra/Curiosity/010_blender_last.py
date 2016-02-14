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
    expNo = "010"
    expInfo = expNo + "_stack_each_feature" 
    _basePath = Config.FolderBasePath + expInfo + Config.osSep
    
    doTestFlag = False
    path = _basePath + expNo + "_train_tobe.csv"
    testPath = _basePath + expNo + "_test_tobe.csv"
    
    
    # 1. read data
    
    clfNameList = []
    clfNameList.append("Extra_Trees")
    clfNameList.append("K_NN")
    clfNameList.append("Random_Forest")
    clfNameList.append("Xgboost")
    clfNameList.append("Logistic_Regression")
    
    featureList = ["event_type", "log_feature", "resource_type", "severity_type"]
    tmpWeightList = [[ 0.24276169,0.02004454,0.00445434,0.71714922,0.0155902 ],[ 0.00310559,0.53881988,0.03416149,0.4052795, 0.01863354],[0.13333333,0.01333333,0.01142857,0.73142857,0.11047619],[ 0.08222222,0.00222222,0.00222222,0.00222222,0.91111111]]

    
    
    
#     for tmpFeature in featureList:
#         dr = DataReader()
#         tmpPath = _basePath + "010_blender_" + tmpFeature + "_train.csv"
#         newX, tmpY =  dr.cvtPathListToDfList(tmpPath, "train") 
#         tmpDf = pd.concat([tmpDf, newX], axis=1)
#         
#     tmpDf.to_csv(tmpOutPath, sep=',', encoding='utf-8')
    
    tmpI, tmpJ =0,0;
    dr = DataReader()
    baseDf, ansY =  dr.cvtPathListToDfList(_basePath+"010_blenderXgboost_train.csv", "train") 


    tmpOutPath = _basePath + "010_train_last_blender.csv" 
    tmpFeatureBlendedAns = pd.DataFrame()
    baseDf = pd.DataFrame()
    tmpDfList =[]
    for tmpClfName in clfNameList:
        dr = DataReader()
        tmpPath = _basePath + "010_"  + "blender" + tmpClfName + "_train.csv"
        newX, tmpY =  dr.cvtPathListToDfList(tmpPath, "train")
        tmpDfList.append(newX)
    
        
    b1 = Blender(clfNameList, tmpDfList, ansY)
    b1.autoFlow(2000, tmpOutPath)
    
    
    # test
    finalWeight = b1._bestParamList
    tmpOutPath = _basePath + "010_test_last_blender.csv" 
    baseDf = pd.DataFrame()
    tmpI =0
    finalTestDf = pd.DataFrame()
    for tmpClfName in clfNameList:
        dr = DataReader()
        tmpPath = _basePath + "010_"  + "blender" + tmpClfName + "_test.csv"
        newX, tmpY =  dr.cvtPathListToDfList(tmpPath, "train")
        tmpWeight = finalWeight[tmpI]
        newX.multiply(tmpWeight)
        if tmpI ==0:
            finalTestDf = newX
        else:
            finalTestDf.add(newX)
    
    finalTestDf.to_csv(tmpOutPath, sep=',', encoding='utf-8')
    musicAlarm()