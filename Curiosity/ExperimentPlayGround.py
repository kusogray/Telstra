'''
Created on Jan 24, 2016

@author: whmou
'''

from Telstra.DataCollector.DataReader import DataReader as dr
from Telstra.DataAnalyzer.ModelFactory import ModelFactory as fab
if __name__ == '__main__':
    
    path = "/Users/whmou/Kaggle/Telstra/train_merge7.csv"
    
    # 1. read data
    dr = dr()
    dr.readInCSV( path, "train")
    
    # 2. run models
    #print dr._trainDataFrame.as_matrix
    fab = fab()
    fab.getRandomForestClf(dr._trainDataFrame, dr._ansDataFrame)
    