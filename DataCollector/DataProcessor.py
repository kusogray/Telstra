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
    
    _eventTypePath = "/Users/whmou/Kaggle/Telstra/log_feature.csv"
    _pathMain = "/Users/whmou/Kaggle/Telstra/train_merge5.csv"
    _outputPathName = "/Users/whmou/Kaggle/Telstra/train_merge6.csv"

    def __init__(self):
        '''
        Constructor
        '''
    def mergeData(self):
        dr = DataReader()
        dr.readInCSV(self._eventTypePath, "train")
        
        tmpColumnPrefix = "resource_type_"
        df = pd.read_csv(self._eventTypePath, header=0, sep=',')
        
        for i in range (1,54):
            tmpColName = tmpColumnPrefix + str(i)
            dr._trainDataFrame[tmpColName] = 0
    
    
        for i1 in range(0, len(dr._trainDataFrame[dr._trainDataFrame.columns[0]] )):
            for i2 in range(0, len(df[df.columns[0]] )):
                tmpMainId = dr._trainDataFrame[dr._trainDataFrame.columns[0]][i1]
                tmpId = df[df.columns[0]][i2]
                if  tmpMainId == tmpId:
                    tmpVal = df[df.columns[1]][i2]
                    tmpColName = tmpColumnPrefix + str(tmpVal)
                    dr._trainDataFrame[tmpColName][i1] = 1
                if tmpMainId > tmpId:
                    break
    
        print dr._trainDataFrame  
    
    
    def sumExist(self):
        dr = DataReader()
        dr.readInCSV(self._pathMain, "train")
        tmpColumnPrefix = "log_feature_"
        df = pd.read_csv(self._eventTypePath, header=0, sep=',')
        
        for i in range (2,3):
            tmpColName = tmpColumnPrefix + str(i)
            dr._trainDataFrame[tmpColName] = 0
        
        tmpLastI2 = 0
        for i1 in range(0, len(dr._trainDataFrame[dr._trainDataFrame.columns[0]] )):

            tmpFlag = False
            for i2 in range(tmpLastI2, len(df[df.columns[0]] )):
                tmpMainId = dr._trainDataFrame[dr._trainDataFrame.columns[0]][i1]
                tmpId = df[df.columns[0]][i2]
                tmpVal= df[df.columns[2]][i2]
                if  tmpMainId == tmpId:
                    tmpFlag = True
                    dr._trainDataFrame[dr._trainDataFrame.columns[6]][i1] +=tmpVal 
                if tmpFlag == True and tmpMainId != tmpId:
                    print "der"
                    tmpLastI2 = i2
                    break
                #print i1, i2
        outDf = pd.concat([dr._ansDataFrame, dr._trainDataFrame], axis=1)
        outDf.to_csv(self._outputPathName, sep=',', encoding='utf-8')  
        #print dr._ansDataFrame
        
if __name__ == '__main__':
    start = time.time()
    
    t = DataProcessor()
    t.sumExist()
    
    end = time.time()
    elapsed = end - start
    print "elapsed:", elapsed , "sec"


