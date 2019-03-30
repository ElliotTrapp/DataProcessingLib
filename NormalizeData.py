'''
NormalizeData.py
Elliot Trapp
18/12/3

Techniques and utlities for normalizing generic data that DO NOT involve modifying 
the data itself, as opposed to transformations that change the data itself.
'''

from sklearn import preprocessing
from scipy import stats
import numpy as np

def MinMax(dataToNormalize, min=0.0, max=1.0):
    """
    @param[in] data a list to be normalized
    @param[in] min target minimum value for scaling
    @param[in] max target maximum value for scaling
    @return rescaled data as a list
    """

    dataToNormalize = np.reshape(dataToNormalize, (-1,1))
    scaler = preprocessing.MinMaxScaler(feature_range=(min,max)).fit(dataToNormalize)
    normalizedData = scaler.transform(dataToNormalize).flatten().tolist()

    return normalizedData

def ZScore(dataToNormalize):
    '''
    NOT WORKING
    Rescales data to an average of zero and a standard deviation of one
    '''

    normalizedData = []
    
    for data in dataToNormalize:
        normalizedData.append(stats.zscore(data)) 

    return normalizedData    