'''
Created on Jan 24, 2016

@author: whmou
'''
import pandas as pd
import time
from Telstra.DataCollector.DataReader import DataReader as DataReader

class DataProcessor(object):
    '''
    classdocs
    '''
    
    _eventTypePath = "/Users/whmou/Kaggle/Telstra/resource_type.csv"
    _pathMain = "/Users/whmou/Kaggle/Telstra/test2.csv"
    _outputPathName = "/Users/whmou/Kaggle/Telstra/test3.csv"

    def __init__(self):
        '''
        Constructor
        '''
    
    def sumExist(self):
        dr = DataReader()
        dr.readInCSV(self._pathMain, "test")
        tmpColumnPrefix = "resource_type_"
        df = pd.read_csv(self._eventTypePath, header=0, sep=',')
        processDf = dr._testDataFrame
        for i in range (1,2):
            tmpColName = tmpColumnPrefix + str(i)
            processDf[tmpColName] = 0
        
        tmpLastI2 = 0
        for i1 in range(0, len(processDf[processDf.columns[0]] )):

            tmpFlag = False
            for i2 in range(tmpLastI2, len(df[df.columns[0]] )):
                tmpMainId = processDf[processDf.columns[0]][i1]
                tmpId = df[df.columns[0]][i2]
                tmpVal= df[df.columns[1]][i2]
                if  tmpMainId == tmpId:
                    tmpFlag = True
                    processDf[processDf.columns[2]][i1] +=1 
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
    
    t = DataProcessor()
    t.sumExist()
    
    end = time.time()
    elapsed = end - start
    print "elapsed:", elapsed , "sec"


