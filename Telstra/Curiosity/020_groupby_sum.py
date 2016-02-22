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

if __name__ == '__main__':
    
    
   # 1. read in data
    expNo = "020"
    expInfo = expNo + "_groupby_sum" 
    _basePath = Config.FolderBasePath + expInfo + Config.osSep
    
    featureList = ["location", "event_type", "resource_type" , "severity_type", "log_feature"]
    ans1List = []
    ans2List = []
#     ansPath = _basePath + "014_ans_array.csv"
#     drAns = DataReader()
#     drAns.readInCSV(ansPath, "train")
#     newY = drAns._ansDataFrame

    tmpPath = _basePath + "train_merge_one_hot.csv"
    dr = DataReader()
    dr.readInCSV(tmpPath, "train")
    newX = dr._trainDataFrame
    newY = dr._ansDataFrame


    fab = ModelFactory()
    #fab._setXgboostTheradToOne = True
    fab._gridSearchFlag = True
    fab._singleModelMail = True
    fab._subFolderName = "groupby_sum"  
    fab._n_iter_search = 1
    fab._expInfo = expInfo
    clf = fab.getXgboostClf(newX, newY)
#     
    tmpPath = _basePath + "test_merge_one_hot" + ".csv"
    dr = DataReader()
    dr.readInCSV(tmpPath, "test")
    newX = dr._testDataFrame
    newX  = xgb.DMatrix(newX)
    tmpOutPath = _basePath + expNo +"_" + "Xgboost_" + "groupby_sum"+ "_ans.csv"
    log(clf.predict(newX))
    outDf = pd.DataFrame(clf.predict(newX))
    outDf.to_csv(tmpOutPath, sep=',', encoding='utf-8')
    musicAlarm()
    