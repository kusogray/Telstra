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
import xgboost as xgb
import numpy as np

if __name__ == '__main__':
    
    
   # 1. read in data
    expNo = "021"
    expInfo = expNo + "_stacking" 
    _basePath = Config.FolderBasePath + expInfo + Config.osSep
    

    tmpPath = _basePath + "train.csv"
    dr = DataReader()
    dr.readInCSV(tmpPath, "train")
    X = dr._trainDataFrame
    Y = dr._ansDataFrame
    ori_X = X
    ori_Y = Y
    
    
    evalDataPercentage = 0.5
    sampleRows = np.random.choice(X.index, len(X)*evalDataPercentage) 
            
    
    train_fold_1  =  X.ix[sampleRows]
    train_fold_label_1 = Y.ix[sampleRows]
    train_fold_2  =  X.drop(sampleRows)
    train_fold_label_2 = Y.drop(sampleRows)
    
#     tmpOutPath = _basePath + expNo +"_" + "fold_1.csv"
#     train_fold_1.to_csv(tmpOutPath, sep=',', encoding='utf-8')
#     
#     tmpOutPath = _basePath + expNo +"_" + "fold_2.csv"
#     train_fold_2.to_csv(tmpOutPath, sep=',', encoding='utf-8')
#     exit()
    
    
    tmpPath = _basePath + "test.csv"
    dr2 = DataReader()
    dr2.readInCSV(tmpPath, "test")
    testX = dr2._testDataFrame
    ori_testX = testX
    sampleRows = np.random.choice(testX.index, len(testX)*evalDataPercentage) 
    
    test_fold_1  =  testX.ix[sampleRows]
    test_fold_2  =  testX.drop(sampleRows)

    clfList = ["xgboost", "rf","extra_tree",]

    
    fab = ModelFactory()
    fab._gridSearchFlag = True
    
    dfUpper = pd.DataFrame()
    dfTestUpper = pd.DataFrame()
    eachClfLoopTimes = 1
    for tmpClfName in clfList:
        for i in range(0,eachClfLoopTimes):
            fab._subFolderName = tmpClfName
            fab._n_iter_search = 1
            fab._expInfo = expInfo
            if  tmpClfName == "rf":
                clf = fab.getRandomForestClf(train_fold_1, train_fold_label_1)
            elif tmpClfName == "knn":
                clf = fab.getKnnClf(train_fold_1, train_fold_label_1)
            elif tmpClfName == "extra_tree":
                clf = fab.getExtraTressClf(train_fold_1, train_fold_label_1)
            elif tmpClfName == "xgboost":
                clf = fab.getXgboostClf(train_fold_1, train_fold_label_1)
            
            if tmpClfName == "xgboost":
                predictResult = clf.predict(xgb.DMatrix(train_fold_2))
                predictTestResult = clf.predict(xgb.DMatrix(test_fold_2))
            else:
                predictResult = clf.predict_proba(train_fold_2)
                predictTestResult = clf.predict_proba(test_fold_2)
            
            
            outFold2 = pd.DataFrame(predictResult)
            outFold2.columns = [tmpClfName+str(i), tmpClfName+str(i+1), tmpClfName+str(i+2) ]
            dfUpper = pd.concat([dfUpper, outFold2], axis=1)
            
            outTestFold2 = pd.DataFrame(predictTestResult)
            outTestFold2.columns = [tmpClfName+str(i), tmpClfName+str(i+1), tmpClfName+str(i+2) ]
            dfTestUpper = pd.concat([dfTestUpper, outTestFold2], axis=1)
        
        
    dfLower = pd.DataFrame()
    dfTestLower = pd.DataFrame()
    for tmpClfName in clfList:
        for i in range(0,eachClfLoopTimes):
            fab._subFolderName = tmpClfName
            fab._n_iter_search = 1
            fab._expInfo = expInfo
            if  tmpClfName == "rf":
                clf = fab.getRandomForestClf(train_fold_2, train_fold_label_2)
            elif tmpClfName == "knn":
                clf = fab.getKnnClf(train_fold_2, train_fold_label_2)
            elif tmpClfName == "extra_tree":
                clf = fab.getExtraTressClf(train_fold_2, train_fold_label_2)
            elif tmpClfName == "xgboost":
                clf = fab.getXgboostClf(train_fold_2, train_fold_label_2)
                
            
            if tmpClfName == "xgboost":
                predictResult = clf.predict(xgb.DMatrix(train_fold_1))
                predictTestResult = clf.predict(xgb.DMatrix(test_fold_1))
            else:
                predictResult = clf.predict_proba(train_fold_1)
                predictTestResult = clf.predict_proba(test_fold_1)
            
            outFold1 = pd.DataFrame(predictResult)
            outFold1.columns = [tmpClfName+str(i), tmpClfName+str(i+1), tmpClfName+str(i+2) ]
            dfLower = pd.concat([dfLower, outFold1], axis=1)    
            
            outTestFold1 = pd.DataFrame(predictTestResult)
            outTestFold1.columns = [tmpClfName+str(i), tmpClfName+str(i+1), tmpClfName+str(i+2) ]
            dfTestLower = pd.concat([dfTestLower, outTestFold1], axis=1)   
        
    mergeDf = dfUpper.append(dfLower)
    mergeTestDf = dfTestUpper.append(dfTestLower)
    
    mergeAns = train_fold_label_2.append(train_fold_label_1)
    
    # Testing
         
    tmpOutPath = _basePath + expNo +"_" + "Xgboost_" + "stacking"+ "_ans.csv"
    
    fab = ModelFactory()
    fab._gridSearchFlag = True
    fab._singleModelMail = True
    fab._subFolderName = "stacking_level2_xgboost"  
    fab._n_iter_search = 1
    fab._expInfo = expInfo
    clf = fab.getXgboostClf(mergeDf, mergeAns)
    
    log(clf.predict(xgb.DMatrix(mergeTestDf)))
    outDf = pd.DataFrame(clf.predict(xgb.DMatrix(mergeTestDf)))
    outDf.to_csv(tmpOutPath, sep=',', encoding='utf-8')
    musicAlarm()
    