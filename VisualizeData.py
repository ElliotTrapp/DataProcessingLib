'''
VisualizeData.py
Elliot Trapp
18/12/3

Utilities for visualizing data throughout the processing and compilation procedure
'''
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import time, datetime
import os
from Libraries.FileManagementLib.FileIO import plot_dir


default_plot_dir = plot_dir

def SavePlot(plot_title, plot_dir=default_plot_dir):
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d')

    plot_title = plot_title.replace(" ","")
    plot_title = plot_title.replace(".", "")
    plot_title = plot_title.replace("\\", "")
    plot_title = plot_title.replace("/", "")

    plot_title = str(plot_dir)+str(st)+'_'+str(plot_title)

    # Don't overwrite old plots
    i = 0
    while os.path.exists('{}_{:d}.jpg'.format(plot_title, i)):
        i += 1
    
    plt.savefig('{}_{:d}.jpg'.format(plot_title, i))

    plt.clf()
    plt.close('all')

def SimplePlot(data_to_plot,plot_title=None,x_label=None,y_label=None):
    fig = plt.figure()
    figure(figsize=(7,7))
    plt.tight_layout()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim([1,len(data_to_plot)+1])
    plt.plot(data_to_plot, 'b-', lw=1)
    plt.grid(True)
    plt.clf()
    plt.close(fig)
    plt.close('all')

def ShowSimplePlot(data_to_plot,plot_title=None,x_label=None,y_label=None):
    fig = plt.figure()
    figure(figsize=(7,7))
    plt.tight_layout()
    plt.title(plot_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim([1,len(data_to_plot)+1])
    plt.plot(data_to_plot, 'b-', lw=1)
    plt.grid(True)
    plt.show()
    plt.clf()
    plt.close(fig)
    plt.close('all')

def SaveSimplePlot(data_to_plot,plot_title=None,x_label=None,y_label=None,plot_dir=default_plot_dir):
    x_lim = float(0.01 * len(data_to_plot))
    x_scale = np.arange(0.0,x_lim,0.01)
    
    fig = plt.figure()
    figure(figsize=(7,7))
    plt.tight_layout()
    plt.title(plot_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    #plt.xlim([1,len(data_to_plot)+1])
    plt.plot(x_scale,data_to_plot, 'b-', lw=1)
    plt.grid(True)
    SavePlot(plot_title=plot_title, plot_dir=plot_dir)
    plt.clf()
    plt.close(fig)
    plt.close('all')

def BeforeAfterPlot(first_data, second_data, x_scale=np.arange(0.0,2.0,0.01), first_title=None, second_title=None, 
    first_x_label=None,first_y_label=None,second_x_label=None,second_y_label=None):
    fig = plt.figure()
    figure(figsize=(15,10))
    # First plot
    plt.subplot(121)
    plt.tight_layout()
    plt.title(first_title)
    plt.xlabel(first_x_label)
    plt.ylabel(first_y_label)
    plt.plot(x_scale, first_data, 'b-', lw=1)
     # Second plot
    plt.subplot(122)
    plt.tight_layout()
    plt.title(second_title)
    plt.xlabel(second_x_label)
    plt.ylabel(second_y_label)
    plt.plot(x_scale,second_data, 'b-', lw=1)
    plt.clf()
    plt.close(fig)
    plt.close('all')
