'''
TransformData.py
Elliot Trapp
18/12/3

Techniques and utlities for cleaning and processing generic data that DO involve modifying 
the data itself, as opposed to normalizations that simply change how the same data is
represented
'''

import numpy as np
import pandas as pd
from sklearn import decomposition
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, normalize, MinMaxScaler
import pywt


def NormalizeData(data_to_norm):
    'Perform min-max normalization on data_to_norm'

    scaler = MinMaxScaler()
    #scaler = MinMaxScaler(feature_range=(-1,1))
    scaled_data = scaler.fit_transform(np.reshape(data_to_norm, (-1,1)))
    
    return scaled_data

def PCA(dataToTransform, n_components=5):
    'Performs Principal Component Analysis on data'

    pca = decomposition.PCA(n_components=n_components)
    transformedData = pca.fit_transform(dataToTransform)

    print("Total variance captured: ", pca.explained_variance_ratio_.sum())
    print("Dimensional distribution of variance: ", pca.explained_variance_ratio_)

    return np.array(transformedData)

def Bin(dataToBin, numBins=1000):
    # Bins the data
        
    box = np.ones(numBins)/numBins
    binned = np.convolve(dataToBin, box, mode='same').flatten().tolist()

    return binned

def WaveletSmooth(data_to_smooth,threshold_ratio=8, waveletType='haar'):
    'Performs a discrete wavelet transform in order to denoise a signal'

    # Decompose the signal
    ca, cd = pywt.dwt(data_to_smooth, waveletType,axis=-1)

    # Create the thresholds
    cat = pywt.threshold(ca, np.std(ca)*threshold_ratio)
    cdt = pywt.threshold(cd, np.std(cd)*threshold_ratio)

    # Reconstruct the signal
    smoothed_data = (pywt.idwt(cat, cdt, waveletType))

    return smoothed_data

def SplitList(list_to_split, new_len):
    'Yield successive num_chunks sized chunks from list_to_split.'
     
    split_list = []

    for element in range(0, len(list_to_split), new_len):
        new_lst = np.array(list_to_split[element:element + new_len])
        split_list.append(new_lst)
    
    return np.array(split_list)

def SlideSampleList(list_to_split, new_len):
    'Yield successive num_chunks sized chunks from list_to_split.'
     
    split_list = []

    for element in range(0, len(list_to_split), 1):
        new_lst = np.array(list_to_split[element:element + new_len])
        split_list.append(new_lst)
    
    return np.array(split_list)

def SampleInput(list_to_sample, sample_len=200):
    
    if len(list_to_sample) < sample_len: print("List too small"); assert(False)
    
    split_list = SplitList(list_to_sample, sample_len)

    assert(len(split_list) <= len(list_to_sample))

    return_lst = []

    for item in split_list:
         # Skip short signals
        if ((len(item) != sample_len)):
            print("raw_data_element with len {0} too small, removing data".format(len(item)))
            continue
        return_lst.append(item)

    
    return np.asarray(return_lst)

def SlideSampleInput(list_to_sample, sample_len=200):
    if len(list_to_sample) < sample_len: print("List too small"); assert(False)
    
    split_list = SlideSampleList(list_to_sample, sample_len)

    assert(len(split_list) <= len(list_to_sample))

    return_lst = []

    for item in split_list:
         # Skip short signals
        if ((len(item) != sample_len)):
            # print("raw_data_element with len {0} too small, padding data".format(len(item)))#, removing data"))
            # item = np.array(PadList(item, max_len=sample_len))
            # assert((len(item) == sample_len))
            continue
        elif ((len(item) > sample_len)):
            print("raw_data_element with len {0} too small, something went wrong".format(len(item)))
            assert(False)
        return_lst.append(item)

    return np.asarray(return_lst)
    
def TrimInput(input_to_trim, beg_num=300, end_num=-500, min_len=2000):

    if len(input_to_trim) < min_len: print("List too small"); print("Length: ",len(input_to_trim)); assert(False)
    
    trimed_list = input_to_trim[beg_num:end_num]

    assert(len(trimed_list) <= len(input_to_trim))

    return trimed_list

def TrimList(list_to_trim, beg_num, end_num):
    'Remove beg_num elements from beginning of list_to_trim and end_num from end'

    return list_to_trim[beg_num:-end_num]

def PadList(list_to_pad, max_len=10000):
    'Pad a list with zeros up to max_len'

    if len(list_to_pad) > max_len:
        print("List too long!"); assert(False)#return list_to_pad[:10000]

    dim_diff = max_len - len(list_to_pad)
    paddedList = np.pad(list_to_pad, (0, dim_diff), 'constant').flatten().tolist()
    
    return paddedList

def TrainTestSplit(train_data, train_labels, test_size=0.25, train_size=None,random_state=None,shuffle=True):
    'Encapsulates sklearns train_test_split'

    train_data, test_data, train_labels, test_labels = train_test_split(train_data, train_labels,
    test_size=test_size, train_size=train_size,random_state=random_state,shuffle=shuffle, )

    return train_data, test_data, train_labels, test_labels

def OneHotEncoderND(dataToEncode, sparse=False,handle_unknown='ignore'):
    'Encapsulates sklearns OneHotEncoder'

    enc = OneHotEncoder(handle_unknown='ignore',sparse=sparse)
    encodedData = enc.fit_transform(dataToEncode)

    return encodedData

def OneHotEncoder1D(dataToEncode, sparse=False,handle_unknown='ignore'):
    'Encapsulates sklearns OneHotEncoder'

    reshaped_data = np.reshape(dataToEncode, (-1, 1))
    encodedData = OneHotEncoderND(reshaped_data)

    return encodedData.tolist()

