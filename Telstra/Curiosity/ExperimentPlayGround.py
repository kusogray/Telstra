'''
Created on Jan 24, 2016

@author: whmou
'''

from Telstra.DataCollector.DataReader import DataReader as DataReader
from Telstra.DataAnalyzer.ModelFactory import ModelFactory as ModelFactory
from sklearn.cross_validation import cross_val_score
from sklearn.externals import joblib
from Telstra.util.CustomLogger import info as log
from Telstra.Resources import Config

import os
import xgboost as xgb
import time  

def exp():
    expInfo = "location_only\\"
    _basePath = Config.FolderBasePath + expInfo
    
    
    doTestFlag = False
    path = _basePath + "train.csv"
    testPath = _basePath + "test10.csv"
   
    # 1. read data
    dr = DataReader()
    dr.readInCSV( path, "train")
    if doTestFlag == True:
        dr.readInCSV(testPath , "test")
    
    # 2. run models
    #print dr._trainDataFrame.as_matrix
    fab = ModelFactory()
    fab._gridSearchFlag = True
    fab._n_iter_search = 10
    fab._expInfo = "location_only"

    X = dr._trainDataFrame
    Y = dr._ansDataFrame
    #fab.getRandomForestClf(X, Y)
    #fab.getAllModels(dr._trainDataFrame, dr._ansDataFrame)
    
#     log( "xgb start")
#     param = {'max_depth':10,  'n_estimators':300 , 'num_class':3, 'learning_rate':0.05, 'objective':'multi:softprob'}
#     num_round = 5
    #gbm = xgb.XGBClassifier(max_depth=10, n_estimators=300, learning_rate=0.05, objective='multi:softprob').fit(dr._trainDataFrame,  dr._ansDataFrame)
    #testResult = gbm.predict_proba(dr._testDataFrame)
    #print testResult
#     gbm = xgb.XGBClassifier(max_depth=10, n_estimators=300, learning_rate=0.05, objective='multi:softprob')
    
#     scores = cross_val_score(rfClf, dr._trainDataFrame,  dr._ansDataFrame, n_jobs = -1)
#     log( "xgboost Validation Precision: ", scores.mean() )
    #xgbCv = xgb.cv(param, xgb.DMatrix(dr._trainDataFrame, dr._ansDataFrame),  num_round, nfold=5,metrics={'error'}, seed = 0)
    #gbTrain = gbm.fit(dr._trainDataFrame,  dr._ansDataFrame)
    #joblib.dump(gbTrain, xgbModelPath)
    #clf = joblib.load( xgbModelPath )
    #clf.predict_proba(dr._testDataFrame)
    #xgb.save(gbm, xgbModelPath)
    #print xgbCv
    #print "xgb end"
    
    #gbm = joblib.load( xgbModelPath )
    #finalClf = gbm
    
    if doTestFlag == True:
        print finalClf.predict_proba(dr._testDataFrame)
    
    
#     featureImportance =[]
#     for i in range(0,len(finalClf.feature_importances_)):
#         if i !=  len(dr._trainDataFrame.columns):  
#             if (dr._trainDataFrame.columns[i]).find("_one_hot") == -1:
#                 featureImportance.append(  [dr._trainDataFrame.columns[i] , finalClf.feature_importances_[i]] )
#     
#     print featureImportance
#     featureImportance.sort(lambda x, y: cmp(x[1], y[1]), reverse=True)
#     print featureImportance 

    if doTestFlag == True:       
        return finalClf.predict_proba(dr._testDataFrame)

if __name__ == '__main__':
    start = time.time()
    
    exp()
    
    end = time.time()
    elapsed = end - start
    log( "exp elapsed:", elapsed , "sec")
    #os.startfile('D:\\123.m4a')
    