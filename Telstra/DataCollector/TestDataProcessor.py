'''
Created on Jan 24, 2016

@author: whmou
'''
import pandas as pd
import time
from Telstra.DataCollector.DataReader import DataReader as DataReader
import os


class TestDataProcessor(object):
    '''
    classdocs
    '''
    _basePath =""
    if os.name == 'nt':
        _basePath = "D:\\Kaggle\\Telstra\\"
    else:
        _basePath = "/Users/whmou/Kaggle/Telstra/"
    
    _typeName= "localtion"
    _eventTypePath = _basePath + _typeName + ".csv"
    _pathMain = _basePath + "test10.csv"
    _outputPathName = _basePath + "test11.csv"
    _mode = "test"

    def __init__(self):
        '''
        Constructor
        '''
    
    def sumExist(self):
        dr = DataReader()
        dr.readInCSV(self._pathMain, self._mode)
        tmpColumnPrefix = self._typeName + "_"
        df = pd.read_csv(self._pathMain, header=0, sep=',')
        if self._mode == "train":
            processDf = dr._trainDataFrame
        else:
            processDf = dr._testDataFrame
            
        for i in range (1,1127):
            tmpColName = tmpColumnPrefix + "one_hot_" + str(i)
            processDf[tmpColName] = 0
        
        tmpLastI2 = 0
        for i1 in range(0, len(processDf[processDf.columns[0]] )):

            tmpFlag = False
            for i2 in range(tmpLastI2, len(df[df.columns[0]] )):
                tmpMainId = processDf[processDf.columns[0]][i1]
                tmpId = df[df.columns[0]][i2]
                tmpVal= df[df.columns[1]][i2]
                #tmpVal2= df[df.columns[2]][i2]
                if  tmpMainId == tmpId:
                    tmpFlag = True
                    processDf[processDf.columns[tmpVal+394]][i1] =1
                if tmpFlag == True and tmpMainId != tmpId:
                    tmpLastI2 = i2
                    break
                #print i1, i2
        #outDf = pd.concat([dr._ansDataFrame, processDf], axis=1)
        outDf = processDf
        outDf.to_csv(self._outputPathName, sep=',', encoding='utf-8')  
        #print dr._ansDataFrame
        
if __name__ == '__main__':
    start = time.time()
    
    t = TestDataProcessor()
    t.sumExist()
    
    end = time.time()
    elapsed = end - start
    print "elapsed:", elapsed , "sec"


