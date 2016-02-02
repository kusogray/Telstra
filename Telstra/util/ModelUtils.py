'''
Created on Jan 31, 2016
author: whmou

Jan 31, 2016     1.0.0     Init.

'''

from sklearn.externals import joblib
from Telstra.util.CustomLogger import info as log
from Telstra.Resources import Config
import os
import time

def getDumpFilePath(clfName, expInfo):

        expInfoFolder = Config.FolderBasePath + expInfo + Config.osSep
        if not os.path.exists(expInfoFolder):
            os.makedirs(expInfoFolder)
        return expInfoFolder + "(" + clfName + ')_(' + str(time.strftime("%Y-%m-%d")) + '_' + str((time.strftime("%H_%M_%S"))) +').model'


def dumpModel(clf, clfName, expInfo):
    
    tmpDumpPath = getDumpFilePath(clfName, expInfo)
    log("Start dump ",clfName, " to " + tmpDumpPath)
    log("Exp info: ",expInfo)
    joblib.dump(clf, tmpDumpPath)
    log("Dump ",clfName, " successfully")
    

def loadModel(modelPath):
    
    log("Start load model: ", modelPath)
    clf = joblib.load( modelPath )
    return clf
     

if __name__ == '__main__':
    log(getDumpFilePath("test", "test"))