'''
CompileData.py
Elliot Trapp
18/12/3

Utilities and functions to compile data from raw Tobii data to data ready to be preprocessed.
Data lifecycle: raw data -> compiled data -> preprocessed data -> training data
'''

import numpy as np
import os,os.path
from pandas import read_csv
from pandas import DataFrame
from Libraries.FileManagementLib.Utilities import ListFullPaths, GetCurrentTimeStamp
import Libraries.FileManagementLib.FileIO as IO


def GetFileMetaDataHelper(metadata, root_dir=IO.raw_train_dir,
                  recursive=False):
    '''Iterates through a directory of raw Tobii data (.csv) and based on the names of the files, assigns metadata and compiles everything into a python dictonary'''

    # Go through each file
    for File in ListFullPaths(root_dir):
        print ("Processing file: ", File)
        
        if (os.path.isfile(File)):
            # If the file is a file
            
            # Write name: {date}_{test_num}_{data_type}
            # Extract data_type

            # Only use gaze data
            if "Gaze" not in File:
                print("Skipping non-gaze data file")
                continue
            if 'Instruction' in File:
                continue

            # Extract eye
            if "Right" in File:
                eye = "Right"
            elif "Left" in File:
                eye = "Left"
            else:
                print("File eye can't be deduced!")
                print("File name:\n", File)
                continue

            # Extract test_type
            if "_FollowSolveSpin" in File:
                # test_type = "FollowSolveSpin"
                # target_val = "High"  
                print("Skipping FollowSolveSpin Test")
                continue          
            elif "_FollowSolve" in File:
                test_type = "FollowSolve"
                target_val = "High"
            elif "_Follow" in File:
                test_type = "Follow"
                target_val = "Low"
            else:
                print("File test_type and target_val can't be deduced!")
                print("File name:\n", File)
                continue

            # Extract subject
            if "noworkload" in File or 'elliottrapp' in File:
                sub = "Sub1"
            elif 'daniel_n' in File:
                sub = 'Sub2'
            elif "daniel" in File:
                sub = "Sub3"
            elif 'aly' in File:
                sub = 'Sub4'
            elif 'jack' in File:
                sub = 'Sub5'
            elif 'test2' in File:
                sub = 'Sub6'
            elif 'test3' in File:
                sub = 'Sub7'
            elif 'test4' in File:
                sub = 'Sub8'
            elif 'test5' in File:
                sub = 'Sub9'
            elif 'gio' in File:
                sub= 'Sub10'
            else:
                print("File subject can't be deduced!")
                print("File name:\n", File)
                assert(False)

            # Extract test_date
            if '180911' in File:
                test_date = '180911'
            elif '180919' in File:
                test_date = '180919'
            elif '181102' in File:
                test_date = '181102'
            elif '181105' in File:
                test_date = '181105'
            elif '181106' in File:
                test_date = '181106'
            elif '181109' in File:
                test_date = '181109'
            else:
                print("File test_type and target_val can't be deduced!")
                print("File name:\n", File)
                assert(False)
        
            assert(test_type is not None)
            assert(sub is not None)
            assert(eye is not None)
            assert(target_val is not None)
            assert(test_date is not None)
            
            metadata.append([File, test_type, test_date, sub, eye, target_val])


        elif (os.path.isdir(File)) and (recursive):
            # If the file is a directory and recursive is set to true
            GetFileMetaDataHelper(metadata, File ,recursive)

    return metadata

def GetFileMetaData(root_dir=IO.raw_train_dir,
                  recursive=False):
    '''Iterates through a directory of raw Tobii data (.csv) and based on the names of the files, assigns metadata and compiles everything into a python dictonary'''

    metadata = []

    return GetFileMetaDataHelper(metadata, root_dir=root_dir, recursive=recursive)

def CompileData(root_dir=IO.raw_train_dir,
                output_dir=IO.compiled_train_dir,
                recursive=False):
    '''
    Converts raw data (seperate disorganized csvs) into compiled data (a single csv containing all fields of information)
    Write file:
    |Source_File_Name|Test_Date|Mental_Workload|Subject|Display|X_Pos|Y_Pos|Start_Time|Left_Pupil_Diam_mm|Right_Pupil_Diam_mm
    '''
      
    st = GetCurrentTimeStamp()
    metadata = GetFileMetaData(root_dir,recursive=recursive)    
    complete_df = DataFrame()

    for file_name, test_type, test_date, subject, eye, target_val in metadata:
        

        matches = (data for data in metadata if data[1] == test_type and data[2] == test_date and data[3] == subject)

        for match in matches:
            if match[4] == "Right":
                right = match
            elif match[4] == "Left":
                left = match
            else:
                print("Failed to find match!")
                assert(False)
                
        l_file = read_csv(left[0])
        r_file = read_csv(right[0],usecols=[4])
        r_file = r_file.rename(index=str, columns={"Pupil Diameter" : "Right_Pupil_Diam_mm"})
        l_file = l_file.rename(index=str, columns={'Pupil Diameter' : 'Left_Pupil_Diam_mm'})
        l_file['Right_Pupil_Diam_mm'] = r_file['Right_Pupil_Diam_mm']

        this_df = DataFrame(l_file)
        this_df["Source_File_Name"] = ([os.path.basename(file_name)] * len(this_df.index))
        this_df["Test_Date"] = ([test_date] * len(this_df.index))
        this_df["Subject"] = ([subject] * len(this_df.index))
        this_df["Mental_Workload"] =  ([target_val] * len(this_df.index))
        complete_df = complete_df.append(this_df)

    # Rename cols
    complete_df['Start_Time_secs'] = complete_df['Start Time (secs)']
    complete_df['X_Pos'] = complete_df['X Pos']
    complete_df['Y_Pos'] = complete_df['Y Pos']
    # Reorder cols
    cols = ['Source_File_Name','Test_Date','Mental_Workload','Subject','Display','X_Pos','Y_Pos','Start_Time_secs','Left_Pupil_Diam_mm','Right_Pupil_Diam_mm']
    complete_df = complete_df[cols]
    # Write to disk
    save_name = output_dir + r'\compileDate={0}_CompiledData.csv'.format(st)
    complete_df.to_csv(save_name,header=True,index=False)