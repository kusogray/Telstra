'''
Created on Jan 24, 2016

@author: whmou
'''
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier as rf

class ModelFactory(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
    
    def getRandomForestClf(self, X, Y):
        clf = rf(n_estimators=300, max_depth=None, min_samples_split=1, random_state=0)
        scores = cross_val_score(clf, X, Y)
        print "Random Forest Validation Precision: ", scores.mean() 
        clf.fit(X, Y)
        return clf
            