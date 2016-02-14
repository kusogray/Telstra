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
    expNo = "013"
    expInfo = expNo + "_data_exploration" 
    _basePath = Config.FolderBasePath + expInfo + Config.osSep
    
    path = _basePath + expNo + "_train_tobe.csv"
    testPath = _basePath + expNo + "_test_asis.csv"
    
    testSortIdPath = Config.FolderBasePath + "test_sort_id.csv"
    trainSortIdPath = _basePath + "train_sort_id.csv"
    
    dr = DataReader()
    dr.readInCSV( path, "train")
    newX, newY = dr._trainDataFrame, dr._ansDataFrame
    
    dr2 = DataReader()
    dr2.readInCSV( testPath, "test")
    #newX = dr2._testDataFrame
    
    dr3 = DataReader()
    dr3.readInCSV( testSortIdPath, "test")
    sortIdDf =dr3._testDataFrame
    
    dr4 = DataReader()
    dr4.readInCSV(trainSortIdPath, "test")
    sortIdDf =dr4._testDataFrame
    
    modelFolder = _basePath + "models" + Config.osSep  + "binary" + Config.osSep
    curModel = "Xgboost"
    modelPath =  modelFolder + str(getMatchNameModelPath(modelFolder, curModel))
    tmpOutPath = _basePath + expNo + "_" +  curModel + "_test_ans.csv"
    tmpClf = loadModel( modelPath)
    log(tmpClf.predict_proba(newX))
    ans = tmpClf.predict_proba(newX)
    
    ansList = []
    for i, tmpAns in enumerate (dr._ansDataFrame):
        if ans[i][tmpAns] < 0.35:
            #log( "id: " + sortIdDf[sortIdDf.columns[0]][i] + ", prob: " + ans[i][tmpAns], ", cate: " + tmpAns)
            log((sortIdDf[sortIdDf.columns[0]][i],ans[i][tmpAns],tmpAns))
            ansList.append((sortIdDf[sortIdDf.columns[0]][i],ans[i][tmpAns],tmpAns))
    
    log (len(ansList))