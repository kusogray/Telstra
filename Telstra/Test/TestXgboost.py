'''
Created on Jan 25, 2016

@author: yu1
'''

import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import numpy as np

if __name__ == '__main__':
    a= {}
    a['a'] = 1
    a['b'] = 3
    a['c'] = 2
    
    print sorted(a.items(), key=lambda x: x[1], reverse = True)[0].key 