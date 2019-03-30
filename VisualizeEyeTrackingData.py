'''
VisualizeEyeTrackingData.py
Elliot Trapp
18/12/3

Utilities and to visualize eye tracking data
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from Libraries.DataProcessingLib.VisualizeData import SavePlot, SimplePlot, BeforeAfterPlot, SaveSimplePlot
from Libraries.FileManagementLib.FileIO import plot_dir

x_label = "Time (sec)"
abs_y_label = "Pupil Diameter (mm)"
norm_y_label = "Normalized Pupil Diameter (0-1)"

default_plot_dir = plot_dir

def BuildPlotTitle(eye,stage,target_str,ID):
    return str('Eye={0}_Stage={1}_Target={2}_ID={3}'.format(eye,stage,target_str,ID))

def SavePreprocessedData(data_to_plot,plot_title="Preprocessed Data", plot_dir=default_plot_dir):
    fig = SimplePlot(data_to_plot,plot_title=plot_title,x_label=x_label,y_label=abs_y_label)
    SavePlot(plot_title=plot_title, plot_dir=default_plot_dir)
    plt.clf()
    plt.close(fig)
    plt.close('all')

def SaveProcessedData(data_to_plot,title="Processed Data", plot_dir=default_plot_dir):
    fig = SimplePlot(data_to_plot,plot_title=plot_title,x_label=x_label,y_label=norm_y_label)
    SavePlot(plot_title=plot_title, plot_dir=default_plot_dir)
    plt.clf()
    plt.close(fig)
    plt.close('all')

def ShowPreprocessedData(data_to_plot,plot_title="Preprocessed Data"):
    fig = SimplePlot(data_to_plot,plot_title=plot_title,x_label=x_label,y_label=abs_y_label)
    plt.show()
    plt.clf()
    plt.close(fig)
    plt.close('all')

def ShowProcessedData(data_to_plot,title="Processed Data"):
    fig = SimplePlot(data_to_plot,plot_title=plot_title,x_label=x_label,y_label=norm_y_label)
    plt.show()
    plt.clf()
    plt.close(fig)
    plt.close('all')

def SaveBeforeAfterPlot(first_data, second_data, x_scale=np.arange(0.0,2.0,0.01), first_title="Unprocessed data",
                        second_title="Processed data",plot_dir=default_plot_dir,
                        plot_title="PupilDataProcessing"):

    fig = BeforeAfterPlot(first_data=first_data, second_data=second_data, x_scale=x_scale, first_title=first_title,
                          second_title=second_title,first_x_label=x_label,first_y_label=abs_y_label,
                          second_x_label=x_label,second_y_label=norm_y_label)
    
    SavePlot(plot_title=plot_title, plot_dir=default_plot_dir)
    plt.close(fig)
    plt.close('all')

def ShowBeforeAfterPlot(first_data, second_data, x_scale=np.arange(0.0,2.0,0.01), first_title="Unprocessed data",
                        second_title="Processed data"):
    fig = BeforeAfterPlot(first_data=first_data, second_data=second_data, x_scale=x_scale, first_title=first_title,
                          second_title=second_title,first_x_label=x_label,first_y_label=abs_y_label,
                          second_x_label=x_label,second_y_label=norm_y_label)
    plt.show()
    plt.clf()
    plt.close(fig)
    plt.close('all')

def ShowDataHeadPlots(data, show_plots, target_str, sample_len):
    VisualizeDataHead(data[:show_plots], target_str, sample_len,show=True)
            
def SaveDataHeadPlots(data, save_plots, target_str, sample_len):
    VisualizeDataHead(data[:save_plots], target_str, sample_len, save=True)

def VisualizeDataHead(raw_data, target, sample_len, show=False, save=False):
    
    to_process, unprocessed = [], []

    for element in raw_data:
        to_process.append(element)
        unprocessed_ele = np.reshape(element, newshape=(2,len(element)))
        unprocessed.append(unprocessed_ele)

    processed = CreateTrainData(to_process, sample_len, 2)
    processed = np.reshape(processed, newshape=(len(processed),2,len(processed[0])))
    
    for index, data in enumerate(processed):

        x_scale = np.arange(0.0,sample_len * 0.01,0.01)

        if (show):
            Viz.ShowBeforeAfterPlot(
            unprocessed[index][0],data[0],
            first_title=BuildPlotTitle('Right', 'Unprocessed', target, index),
            second_title=BuildPlotTitle('Right', 'Processed', target, index))

            Viz.ShowBeforeAfterPlot(
            unprocessed[index][0],data[0],
            first_title=BuildPlotTitle('Left', 'Unprocessed', target, index),
            second_title=BuildPlotTitle('Left', 'Processed', target, index)) 

        if (save):
            Viz.SaveBeforeAfterPlot(
            unprocessed[index][0],data[0],x_scale,
            first_title=BuildPlotTitle('Right', 'Unprocessed', target, index),
            second_title=BuildPlotTitle('Right', 'Processed', target, index),
            plot_title=BuildPlotTitle('Right', 'Before/After', target, index))

            Viz.SaveBeforeAfterPlot(
            unprocessed[index][0],data[0],x_scale,
            first_title=BuildPlotTitle('Left', 'Unprocessed', target, index),
            second_title=BuildPlotTitle('Left', 'Processed', target, index),
            plot_title=BuildPlotTitle('Left', 'Before/After', target, index))
        

def PlotBothEyeData(left_data, right_data, stage, target_str, ID):
    PlotSingleEyeData(left_data, eye='Left', stage=stage, target_str=target_str, ID=ID)
    PlotSingleEyeData(right_data, eye='Right', stage=stage, target_str=target_str, ID=ID)

def PlotSingleEyeData(data, eye, stage, target_str, ID):

    plot_dir = default_plot_dir + r'/{0}/{1}/'.format(stage, target_str)

    if (stage == 'Processed'):
        y_label = 'Pupil Diameter (Normalized between 0-1)'
        # x_scale= np.arange(0.0, 8.0 ,0.01)
    elif (stage == 'Unprocessed'):
        y_label = 'Pupil Diameter (mm)'
        # x_scale= np.arange(0.0, float(len(data) * 0.01) ,0.01)
    else:
        assert(False)

    SaveSimplePlot(data,
        plot_title=BuildPlotTitle(eye,stage,target_str,ID),
        plot_dir=plot_dir,
        x_label='Time (Sec)',
        y_label=y_label)
    