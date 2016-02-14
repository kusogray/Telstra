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

    
    tmpOutPath = _basePath + "010_test_tobe.csv"
    
#     for tmpFeature in featureList:
#         dr = DataReader()
#         tmpPath = _basePath + "010_blender_" + tmpFeature + "_train.csv"
#         newX, tmpY =  dr.cvtPathListToDfList(tmpPath, "train") 
#         tmpDf = pd.concat([tmpDf, newX], axis=1)
#         
#     tmpDf.to_csv(tmpOutPath, sep=',', encoding='utf-8')
    
    tmpI, tmpJ =0,0;
    dr = DataReader()
    baseDf, tmpY =  dr.cvtPathListToDfList(_basePath+"010_test_asis.csv", "test") 
    for tmpFeature in featureList:
        outputPath = _basePath + expNo + "_blender_" + tmpFeature + "_test.csv"
        #ansPath = _basePath + "010_Extra_Trees_stack_event_type.csv"
        #dr = DataReader()
        #tmpX, ansY = dr.cvtPathListToDfList(ansPath, "train")
        #tmpDfList = []        
    
        tmpFeatureBlendedAns = pd.DataFrame()
        for tmpClfName in clfNameList:
            dr = DataReader()
            tmpPath = _basePath + "010_" + tmpClfName + "_stack_" + tmpFeature + "_test.csv"
            newX, tmpY =  dr.cvtPathListToDfList(tmpPath, "train")
            tmpWight = tmpWeightList[tmpI][tmpJ]
            
            newX = newX.multiply(tmpWight)
            if tmpJ ==0:
                tmpFeatureBlendedAns = newX
            else:
                tmpFeatureBlendedAns = tmpFeatureBlendedAns.add(newX)
            tmpJ +=1
            
        baseDf = pd.concat([baseDf, tmpFeatureBlendedAns],axis =1)
        tmpI +=1
        tmpJ =0
        #b1 = Blender(clfNameList, tmpDfList, ansY)
        #b1.autoFlow(1000, outputPath)
    baseDf.to_csv(tmpOutPath, sep=',', encoding='utf-8')
    
    
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
    
    