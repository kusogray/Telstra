'''
Created on Jan 24, 2016

@author: whmou
'''

from Telstra.DataCollector.DataReader import DataReader as DataReader
from Telstra.DataAnalyzer.ModelFactory import ModelFactory as ModelFactory
import os

def exp():
    _basePath =""
    if os.name == 'nt':
        _basePath = "D:\\Kaggle\\Telstra\\"
    else:
        _basePath = "/Users/whmou/Kaggle/Telstra/"
    
    doTestFlag = False
    path = _basePath + "train_merge10.csv"
    testPath = _basePath + "test9.csv"
    # 1. read data
    dr = DataReader()
    dr.readInCSV( path, "train")
    dr.readInCSV(testPath , "test")
    
    # 2. run models
    #print dr._trainDataFrame.as_matrix
    fab = ModelFactory()
    rfClf = fab.getRandomForestClf(dr._trainDataFrame, dr._ansDataFrame)
    
    if doTestFlag == True:
        print rfClf.predict_proba(dr._testDataFrame)
    
    
    featureImportance =[]
    for i in range(0,len(rfClf.feature_importances_)):
        if i !=  len(dr._trainDataFrame.columns):  
            if (dr._trainDataFrame.columns[i]).find("_one_hot") == -1:
                featureImportance.append(  [dr._trainDataFrame.columns[i] , rfClf.feature_importances_[i]] )
    
    print featureImportance
    featureImportance.sort(lambda x, y: cmp(x[1], y[1]), reverse=True)
    print featureImportance 

    if doTestFlag == True:       
        return rfClf.predict_proba(dr._testDataFrame)

if __name__ == '__main__':
    exp()
    