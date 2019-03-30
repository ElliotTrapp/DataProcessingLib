'''
TransformEyeTrackingData.py
Elliot Trapp
18/12/3

Techniques and utlities for cleaning and processing eyetracking data that DO involve modifying 
the data itself, as opposed to normalizations that simply change how the same data is
represented
'''

import numpy as np
import matlab.engine
import Libraries.DataProcessingLib.TransformData as GenericTransform

def ProcessRawEyeData(raw_data_element,sample_len,n_dim):
    """
    Prepare raw data for training
    """
    train_data_element = []

    raw_data_element = np.reshape(raw_data_element, newshape=(n_dim,len(raw_data_element)))
    assert(len(raw_data_element) == n_dim)

    for data_type in raw_data_element:
        assert(len(data_type) == sample_len)

        # Process left, right, and vergence data
        # Deblink and smooth again
        deblinked_data = DeblinkPupilList(data_type)

        # Smooth the data using wavelets
        
        smoothed_data = GenericTransform.WaveletSmooth(deblinked_data)
       
       # Normalize the data between 0 and 1
       ## norm_data = GenericTransform.NormalizeData(smoothed_data)
       

        train_data_element.append(np.array(smoothed_data))

    assert(len(train_data_element) == n_dim)

    train_data_element = np.asarray(train_data_element)
    train_data_element = np.reshape(train_data_element, newshape=(len(train_data_element[0]),n_dim))

    return train_data_element

def RemoveOutliersFromPupilDiameters(pupil_list, timeData):
    # Takes in a list of pupil diameters and runs a MATLAB script to remove outliers

    # Convert to MATLAB numeric array
    MATLAB_pupilDiameters = matlab.double(pupil_list)
    MATLAB_timestamps = matlab.double(timeData)

    eng = matlab.engine.start_matlab()
    eng.cd(r'.\MATLAB')
    completeResults = eng.RemoveOutliers(MATLAB_pupilDiameters, MATLAB_timestamps)

    deoutlieredList = []

    [deoutlieredList.append(float(i[0])) for i in completeResults]
    
    return deoutlieredList

def DeblinkPupilList(pupil_list, smoothData=True,minThreshold=0.3):
    # Takes in a list of pupil diameters and runs a MATLAB script to linearly interpolate
    # through any blinks present and optionally smooth the data

    # Convert to MATLAB numeric array
    MATLAB_inputList = matlab.double(pupil_list.flatten().tolist())

    eng = matlab.engine.start_matlab()
    eng.cd(r'.\MATLAB')

    # Interpolate and smooth blinks

    """""
    % RemoveBlinks.m Inputs:
    %   data is a vector of pupillary data in millimeters
    %     (no default - necessary input)
    %   graphics is whether or not to plot output
    %     (defaults to 0)
    %   lrtask is whether or not the task is a light-reflex task.
    %     if lrtask=0 it expects a task with relatively
    %         slow responses (i.e., no light reflex) and is thus
    %         more conservative.
    %     if lrtask =1, it expects a task with relatively
    %         quick light-reflex responses
    %     (defaults to 0)
    %   manualblinks is a vector of samples that are known to be blinks
    %    (defaults to empty)
    %   lowthresh is a milimeter threshold below which data is assumed to be blinks
    %    (defaults to .1)
    """""
    deblinked_data = eng.RemoveBlinks(MATLAB_inputList,0,0,matlab.double(),minThreshold)

    deblinked_list = []

    if (smoothData):
        [deblinked_list.append(float(i[0])) for i in deblinked_data["NoBlinks"]]
    else:
        [deblinked_list.append(float(i)) for i in deblinked_data["NoBlinksUnsmoothed"][0]]

    return deblinked_list
