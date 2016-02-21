'''
Created on Feb 1, 2016
author: whmou

Feb 1, 2016     1.0.0     Init.

'''
import sys
#print(sys.path)
sys.path.append("D:\\workspace\\Telstra\\")
sys.path.append("D:\\WinPython-64bit-2.7.10.3\\python-2.7.10.amd64\\Lib\\site-packages\\")

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
import scipy as sp
import numpy as np
import xgboost as xgb
from time import gmtime, strftime
from Telstra.util.ModelUtils import *

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

    tmpPath = _basePath + "train_2.csv"
    dr = DataReader()
    dr.readInCSV(tmpPath, "train")
    newX = dr._trainDataFrame
    newY = dr._ansDataFrame
    X = newX
    Y= newY
    evalDataPercentage = 0.1
    
 
    fab = ModelFactory()
    #fab._setXgboostTheradToOne = True
    fab._gridSearchFlag = True
    fab._singleModelMail = True
    fab._subFolderName = "testXgboost8"  
    fab._n_iter_search = 1
    fab._expInfo = expInfo
    clf = fab.getXgboostClf(newX, newY)
#     
    tmpPath = _basePath + "test_2.csv"
    dr = DataReader()
    dr.readInCSV(tmpPath, "test")
    newX = dr._testDataFrame
    newY = dr._ansDataFrame
    newX  = xgb.DMatrix(newX)
    #print clf.predict(newX)
    tmpOutPath = _basePath + expNo +"_" + "Xgboost" + "_testXgboost8_ans.csv"
    log(clf.predict(newX))
    outDf = pd.DataFrame(clf.predict(newX))
    outDf.to_csv(tmpOutPath, sep=',', encoding='utf-8')
    musicAlarm()
    
        
#     sampleRows = np.random.choice(X.index, len(X)*evalDataPercentage) 
#     
#     print  X.ix[sampleRows]
#     exit()
#     dtest  = xgb.DMatrix( X.ix[sampleRows], label=Y.ix[sampleRows])
#     dtrain  =  xgb.DMatrix( X.drop(sampleRows), label=Y.drop(sampleRows))
#     
#     print strftime("%Y-%m-%d %H:%M:%S", gmtime())
#     
#     #dtrain  = xgb.DMatrix( newX[:5000], label=newY[:5000])
#     #dtest  =  xgb.DMatrix( newX[5001:], label=newY[5001:])
#     
#     
#     
#     evallist  = [(dtest,'eval'), (dtrain,'train')]
#     
#     param = {'bst:max_depth':12, 'bst:eta':0.1, 'silent':1, 'eval_metric':'mlogloss', 'num_class':3 , 'objective':'multi:softprob' }
#     param['nthread'] = 4
#     #param['eval_metric'] = 'auc'
#     plst = param.items()
#     num_round = 50
#     print plst
#     bst = xgb.train( plst, dtrain, num_round , evallist)
#     joblib.dump(bst, "F:\\test.model")
#     clf = joblib.load( "F:\\test.model" )
#     print clf.predict(dtest)
# 
#     
#     
#     
#     #bst = xgb.XGBClassifier(max_depth=8, n_estimators=95, learning_rate=0.15,nthread=4, objective='multi:softprob')
#     #bst.train(newX,newY)
#     print strftime("%Y-%m-%d %H:%M:%S", gmtime())
#     
