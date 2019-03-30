'''
ProcessData.py
Elliot Trapp
18/12/3

Primary logic for processing data (compiled data -> training data)
'''

import numpy as np
from pandas import read_csv
from pandas import DataFrame
import pandas as pd
from Libraries.FileManagementLib.Utilities import GetCurrentTimeStamp
import Libraries.FileManagementLib.FileIO as IO
from Libraries.DataProcessingLib.TransformEyeTrackingData import ProcessRawEyeData, DeblinkPupilList
from Libraries.DataProcessingLib.VisualizeEyeTrackingData import PlotBothEyeData
import Libraries.DataProcessingLib.TransformData as Trans


# def CreateTrainData(raw_data, sample_len, n_dim):
#     '''
#     Takes a single 
#     '''
#     # Prepare lists that will be built and returned
#     train_data = []    
#     # Preprocess all raw data and append it to training data list
#     for raw_data_element in raw_data:
        
#         # Skip short signals
#         if ((len(raw_data_element) != sample_len)):
#             print("raw_data_element with len {0} too small, removing data".format(len(raw_data_element)))
#             continue
#         train_data_element = ProcessRawEyeData(raw_data_element, sample_len, n_dim)
#         if (np.amax(raw_data_element) < 0.97):
#             print("raw_data_element with max {0} too small, removing data".format(max(raw_data_element)))
#             continue

#         train_data.append(train_data_element)

#     # return processed data and labels
#     return np.asarray(train_data)


def GetGroupTarget(group):
    '''Gets the target label for a specific group'''
    
    if 'Low' in group['Mental_Workload'].any():
        target = 0
        target_str = 'Low'
    # elif 'Medium' in group['Mental_Workload'].any():
    #     target = 1
    #     target_str = 'Medium'
    elif 'High' in group['Mental_Workload'].any():
        target = 1
        target_str = 'High'
    else:
        print("Failed to get target val")
        print(r'group[\'Mental_Workload\']:\n',group['Mental_Workload'])
        assert(False)

    return target, target_str

def ProcessData(
    load_file= str(IO.compiled_train_dir + r'\compileDate=181106_CompiledData.csv'),
    save_file=IO.postprocessed_train_dir,
    save_plots=True,
    sample_len=200,
    n_dim=2,
    normalize=True):
    '''Primary logic for processing data. Takes compiled data (.csv) and produces 4 files and saves them to save_file. They are:
    A csv file containing the training data
    A csv file containing the training labels
    An npy file containing the training data
    An npy file containing the training labels

    Both csv and npy formats are produced because csv is human readable and the npy is much faster to load
    '''

    train_labels = []
    train_data = []

    complete_df = read_csv(str(load_file),
    usecols=['Mental_Workload','Source_File_Name','Start_Time_secs','Left_Pupil_Diam_mm','Right_Pupil_Diam_mm'])

    test_groups = complete_df.groupby('Source_File_Name')['Mental_Workload','Start_Time_secs','Left_Pupil_Diam_mm','Right_Pupil_Diam_mm']
    
    ID = 0
    # Interate through each test
    for test, group in test_groups:
       
        target, target_str = GetGroupTarget(group)

        data = group[['Left_Pupil_Diam_mm','Right_Pupil_Diam_mm']]

        left_data = data[['Left_Pupil_Diam_mm']].values
        right_data = data[['Right_Pupil_Diam_mm']].values

        # Plot Unprocessed data
        if (save_plots):
            PlotBothEyeData(left_data, right_data, 'Unprocessed', target_str, ID)

        # Trim
        min_len = 2000
        if len(data) < min_len: print("List too small"); print("Length: ",len(data)); continue
        data = Trans.TrimInput(data)

        left_data = data[['Left_Pupil_Diam_mm']].values
        right_data = data[['Right_Pupil_Diam_mm']].values

        if (normalize):
            left_data = Trans.NormalizeData(left_data)
            right_data = Trans.NormalizeData(right_data)

        # Deblink data
        left_data = DeblinkPupilList(left_data)
        right_data = DeblinkPupilList(right_data)

        left_data = Trans.WaveletSmooth(left_data)
        right_data = Trans.WaveletSmooth(right_data)

        assert(len(left_data == len(right_data)))

        # Slide along data
        left_data = Trans.SlideSampleInput(left_data, sample_len)
        right_data = Trans.SlideSampleInput(right_data, sample_len)

        assert(len(left_data == len(right_data)))

        # Plot processed data
        if (save_plots):
            #PlotBothEyeData(left_data, right_data, 'Processed', target_str, ID)
            PlotBothEyeData(left_data[0], right_data[0], 'Processed', target_str, ID)
            PlotBothEyeData(left_data[1], right_data[1], 'Processed', target_str, ID)
            PlotBothEyeData(left_data[2], right_data[2], 'Processed', target_str, ID)

            PlotBothEyeData(left_data[-3], right_data[0], 'Processed', target_str, ID)
            PlotBothEyeData(left_data[-2], right_data[1], 'Processed', target_str, ID)
            PlotBothEyeData(left_data[-1], right_data[2], 'Processed', target_str, ID)

        data = []
        data.append(np.array(left_data))
        data.append(np.array(right_data))
        
        data = np.asarray(data)

        assert(len(data) == n_dim)

        print('before reshape:',data.shape)

        data = np.reshape(data, newshape=(len(left_data),sample_len,n_dim))

        #print('after reshape:',data)
        print('after reshape:',data.shape)

        # Extend to the set of data and labels
        train_data.extend(data)
        train_labels.extend([target]* len(data))

        print('train_data:',np.array(train_data).shape)

        ID += 1

    train_data = np.array(train_data)
    train_labels = np.array(train_labels)

    print("train_data Len:\n",len(train_data))
    print("train_data Shape:\n",train_data.shape)
    print("labels Len:\n",len(train_labels))

    SaveProcessedData(save_file, train_data, train_labels, sample_len)


    return train_data, train_labels

def GetFileNames(out_dir, sample_len):
    stamp = GetCurrentTimeStamp()

    data_csv_file_name = str(out_dir + r'\\{0}_sample_len=({1})_train_data.csv'.format(stamp, sample_len))
    label_csv_file_name = str(out_dir + r'\\{0}_sample_len=({1})_train_labels.csv'.format(stamp, sample_len))

    data_npy_file_name = str(out_dir + r'\\{0}_sample_len=({1})_train_data.npu'.format(stamp, sample_len))
    label_npy_file_name = str(out_dir + r'\\{0}_sample_len=({1})_train_labels.npy'.format(stamp, sample_len))

    return data_csv_file_name, label_csv_file_name, data_npy_file_name, label_npy_file_name

def SaveProcessedData(out_dir, train_data, train_labels, sample_len):
    d_csv, l_csv, d_npy, l_npy = GetFileNames(out_dir, sample_len)

    # csv
    IO.Write3DArray(out_filename=d_csv, out_data=train_data)
    IO.WriteCSV(out_filename=l_csv, out_data=train_labels)

    # npy
    np.save(file = d_npy, arr=train_data)
    np.save(file = l_npy, arr=train_labels)
