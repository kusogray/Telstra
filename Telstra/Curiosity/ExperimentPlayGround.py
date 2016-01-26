'''
Created on Jan 24, 2016

@author: whmou
'''

from Telstra.DataCollector.DataReader import DataReader as DataReader
from Telstra.DataAnalyzer.ModelFactory import ModelFactory as ModelFactory
import os
import xgboost as xgb
import time  

def exp():
    _basePath =""
    if os.name == 'nt':
        _basePath = "D:\\Kaggle\\Telstra\\"
    else:
        _basePath = "/Users/whmou/Kaggle/Telstra/"
    
    doTestFlag = True
    path = _basePath + "train_merge10.csv"
    testPath = _basePath + "test10.csv"
    # 1. read data
    dr = DataReader()
    dr.readInCSV( path, "train")
    dr.readInCSV(testPath , "test")
    
    # 2. run models
    #print dr._trainDataFrame.as_matrix
    fab = ModelFactory()
    #rfClf = fab.getRandomForestClf(dr._trainDataFrame, dr._ansDataFrame)
    
    print "xgb start"
    param = {'max_depth':2}
    num_round = 2
    gbm = xgb.XGBClassifier(max_depth=7, n_estimators=350, learning_rate=0.05, objective='multi:softprob').fit(dr._trainDataFrame,  dr._ansDataFrame)
    testResult = gbm.predict_proba(dr._testDataFrame)
    print testResult
    #xgbCv = xgb.cv(param, xgb.DMatrix(dr._trainDataFrame, dr._ansDataFrame), num_round, nfold=5,
    #   metrics={'error'}, seed = 0)
    
    #print xgbCv
    print "xgb end"
    
    finalClf = gbm
    
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
    print "exp elapsed:", elapsed , "sec"
    