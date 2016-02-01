'''
Created on Jan 24, 2016

@author: whmou
'''
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV

from Telstra.util.CustomLogger import info as log
from Telstra.util.CustomLogger import mail
from Telstra.util.CustomLogger import musicAlarm
from Telstra.util.ModelUtils import *
from sklearn.datasets import load_digits

import numpy as np
from operator import itemgetter
import time
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_randf
from sklearn.externals import joblib
import os

#http://docs.scipy.org/doc/numpy/reference/routines.random.html

from sklearn.ensemble import RandomForestClassifier as rf
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

class ModelFactory(object):
    '''
    classdocs
    '''

    _gridSearchFlag = False
    _n_iter_search = 1
    _expInfo = "ExpInfo"
    
    _bestScoreDict = {}
    _bestClf = {}   # available only when self._gridSearchFlag is True
    _basicClf = {}  # when self._gridSearchFlag is True, basic = best  
    
    def __init__(self):
        '''
        Constructor
        '''
    
    
    def getAllModels(self, X, Y):
        
        log("GetAllModels start with iteration numbers: " , self._n_iter_search)
        start = time.time()
        self._basicClf["Xgboost"] = self.getXgboostClf(X, Y)
        self._basicClf["Random_Forest"] = self.getRandomForestClf(X, Y)
        self._basicClf["Extra_Trees"] = self.getExtraTressClf(X, Y)
        self._basicClf["K_NN"] = self.getKnnClf(X, Y)
        self._basicClf["Logistic_Regression"] = self.getLogisticRegressionClf(X, Y)
        self._basicClf["Naive_Bayes"] = self.getNaiveBayesClf(X, Y)
        log("GetAllModels cost: " , time.time() - start , " sec")
        log(sorted(self._bestScoreDict.items(), key=lambda x: x[1] , reverse=True) )
        mail(self._expInfo, sorted(self._bestScoreDict.items(), key=lambda x: x[1] , reverse=True) )
        log("GetAllModels end with iteration numbers: " , self._n_iter_search)

    # Utility function to report best scores
    def report(self, grid_scores, clfName, n_top=3):
        top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
        bestParameters = {}
        for i, score in enumerate(top_scores):
            log("Model with rank: {0}".format(i + 1))
            log("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  score.mean_validation_score,
                  np.std(score.cv_validation_scores)))
            log("Parameters: {0}".format(score.parameters))
            if i == 0:
                self._bestScoreDict[clfName] = score.mean_validation_score
            print("")
        return bestParameters
    
    
    
        
    def doRandomSearch(self, clfName, clf, param_dist, X, Y):
        start = time.time()
        random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                               n_iter=self._n_iter_search)
        
        random_search.fit(X, Y)
        log(clfName + " randomized search cost: " , time.time() - start , " sec")
        self.report(random_search.grid_scores_, clfName)
        self._bestClf[clfName] = random_search.best_estimator_
        dumpModel(random_search.best_estimator_, clfName, self._expInfo)
        return random_search.best_estimator_
    
    # # 1. Random Forest
    def getRandomForestClf(self, X, Y):
        clfName = "Random_Forest"
        ## http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        clf = rf(n_estimators=300, max_depth=None, min_samples_split=1, random_state=0)
        
        if self._gridSearchFlag == True:
            log(clfName + " start searching param...")
            
            param_dist = {
                          "max_depth": sp_randint(4, 8),
                          "max_features": sp_randf(0,1),
                          "min_samples_split": sp_randint(1, 11),
                          "min_samples_leaf": sp_randint(1, 11),
                          "bootstrap": [True, True],
                          "criterion": ["gini", "entropy"], 
                          "oob_score":[True, True],
                          "n_estimators" : sp_randint(100, 300),
                          }
            
            clf = self.doRandomSearch(clfName, clf, param_dist, X, Y)
            
            
        return clf
    
    
    # # 2.xgboost
    def getXgboostClf(self, X, Y):
        clfName = "Xgboost"
        
        ## https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
        clf = xgb.XGBClassifier(
                                nthread=4,
                                max_depth=12,
                                subsample=0.5,
                                colsample_bytree=1.0, 
                                objective='multi:softprob')
        
        if self._gridSearchFlag == True:
            log(clfName + " start searching param...")
            
            param_dist = {
                          "objective": ['multi:softprob', 'multi:softprob'],
                          "learning_rate": sp_randf(0,1),
                          "gamma": sp_randint(0, 5),
                          "max_depth": sp_randint(4, 10),
                          "min_child_weight": sp_randint(1, 5),
                          "max_delta_step": sp_randint(1, 10),
                          "subsample":sp_randf(0,1),
                          #"colsample_bytree " : sp_randf(0,1),
                          #"num_boost_round": sp_randint(40, 80),
                          #"num_class": [len(set(Y)), len(set(Y))],
                          }
            
            clf = self.doRandomSearch(clfName, clf, param_dist, X, Y)
        
        #joblib.dump(clf, xgbModelPath)    
        return clf

    # # 3.Extra Trees
    def getExtraTressClf(self, X, Y):
        clfName = "Extra_Trees"
        
        ## http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
        clf = ExtraTreesClassifier(
                                n_estimators=10, 
                                criterion='gini', 
                                max_depth=None, 
                                min_samples_split=2, 
                                min_samples_leaf=1, 
                                min_weight_fraction_leaf=0.0, 
                                max_features='auto', 
                                max_leaf_nodes=None, 
                                bootstrap=False, 
                                oob_score=False, 
                                n_jobs=1, 
                                random_state=None, 
                                verbose=0, 
                                warm_start=False, 
                                class_weight=None)
        
        if self._gridSearchFlag == True:
            log(clfName + " start searching param...")
            
            param_dist = {
                          "max_depth": sp_randint(4, 8),
                          "max_features": sp_randint(1, 11),
                          "min_samples_split": sp_randint(1, 11),
                          "min_samples_leaf": sp_randint(1, 11),
                          "bootstrap": [True, True],
                          "criterion": ["gini", "entropy"], 
                          "oob_score":[True, True],
                          "n_estimators" : sp_randint(100, 300),
                          }
            
            clf = self.doRandomSearch(clfName, clf, param_dist, X, Y)
            
        return clf
    
    
    # # 4.KNN
    def getKnnClf(self, X, Y):
        clfName = "K_NN"
        
        ## http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
        clf = KNeighborsClassifier(
                                n_neighbors=5, 
                                weights='uniform', 
                                algorithm='auto', 
                                leaf_size=30, 
                                p=2, 
                                metric='minkowski', 
                                metric_params=None, 

                                )
        
        if self._gridSearchFlag == True:
            log(clfName + " start searching param...")
            
            param_dist = {
                          "n_neighbors": sp_randint(4, 8),
                          "weights": ['uniform', 'uniform'],
                          "leaf_size": sp_randint(30, 60),
                          "algorithm": ['auto', 'auto'],
                          }
            
            clf = self.doRandomSearch(clfName, clf, param_dist, X, Y)
            
        return clf
    
    # # 5.Logistic Regression
    def getLogisticRegressionClf(self, X, Y):
        clfName = "Logistic_Regression"
        
        ## http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        clf = LogisticRegression(
                                penalty='l2', 
                                dual=False, 
                                tol=0.0001, 
                                C=1.0, 
                                fit_intercept=True, 
                                intercept_scaling=1, 
                                class_weight=None, 
                                random_state=None, 
                                solver='liblinear', 
                                max_iter=100, 
                                multi_class='ovr', 
                                verbose=0, 


                                )
        
        if self._gridSearchFlag == True:
            log(clfName + " start searching param...")
            
            param_dist = {
                          "penalty": ['l1', 'l2'],
                          "C": sp_randf(1.0,3.0),
                          "solver": ['newton-cg', 'lbfgs', 'liblinear'],
                          }
            
            clf = self.doRandomSearch(clfName, clf, param_dist, X, Y)
            
        return clf
    
    # # 6. Naive Bayes
    def getNaiveBayesClf(self, X, Y):
        clfName = "Naive_Bayes"
        
        ## http://scikit-learn.org/stable/modules/naive_bayes.html#gaussian-naive-bayes
        clf = GaussianNB()
        clf = clf.fit(X, Y)
        scores = cross_val_score(clf, X,  Y )
        log( clfName + " Cross Validation Precision: ", scores.mean() )
        self._bestScoreDict[clfName] = scores.mean()
            
        return clf
    
if __name__ == '__main__':
    fab = ModelFactory()
    fab._gridSearchFlag = True
    
    #log (sp_randf)
    
    digits = load_digits()
    X, Y = digits.data, digits.target
#     clf = fab.getNaiveBayesClf(X, Y)
#     clf2 = fab.getKnnClf(X, Y)
    #clf3 = fab.getRandomForestClf(X, Y)
#     x= clf.predict_proba(X)
#     log( x)
    #log(fab._bestScoreDict)
#     #log(fab._bestClf)
#     log( fab._bestClf['Random Forest'].predict_proba(X))
    fab.getAllModels(X, Y)
    #log(sorted(fab._bestScoreDict.items(), key=lambda x: x[1] , reverse=True) )
    #log(fab._bestClf['Random Forest'].predict_proba(X))
    #dumpModel(clf3, "Random_Forest", "ExpTest")
    #log("haha")
    #log(getDumpFilePath( "Random_Forest", "haha Tets"))      
    #musicAlarm()
