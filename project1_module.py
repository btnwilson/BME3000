# -*- coding: utf-8 -*-
"""
Ben Wilson, Thomas Bausman
BME3000
Project 1
This file contains functions used in the processing and plotting of ecg data. 
It contains 6 functions, to load the data, plot the raw data, plot the events,
extract trials, plot the mean beats and save important data. 
"""
# import modules 
import numpy as np
import matplotlib.pyplot as plt
import random
# %% Part 1
def load_data(input_file):
    '''
    This function loads data from a .npz file with the path passed as a 
    parameter. 

    Parameters
    ----------
    file_name : str 
        Reltative file path of .npz file storing data. 

    Returns
    -------
    ecg_voltage : ndarray
        1D array of voltages for every sample time as floats. 
    frequency : ndarray
        The frequency in Hz for sample rate as an integer. 
    labeled_samples_indices : ndarray
        A 1d array of integers with the index values of the expert provided annotations. 
    labeled_symbols : ndarray
        A 1d array of strings the same size of labeled_sample_indices with the markers
        corresponding to the index values. 
    subject_id : ndarray
        Contains a string for the patient ID. 
    electrode : ndarray
        contains a string annotation of which electrode the ecg view is 
        taken from. 
    units : ndarray
        contains a string of the units for the ecg voltages. 

    '''
    # open data and load file
    data = np.load(input_file)
    print('Data in file:')
    for file in data.files:
        print(file + ',')
    # store data to variables to be returned. 
    ecg_voltage = data['ecg_voltage']
    electrode = data['electrode']
    frequency = data['fs']
    labeled_samples_indices = data['label_samples']
    labeled_symbols = data['label_symbols']
    subject_id = data['subject_id']
    units = data['units']
    
    # return extracted values
    return ecg_voltage, frequency, labeled_samples_indices, labeled_symbols, subject_id, electrode, units
    
# %% Part 2  
def plot_raw_data(signal_voltage, signal_time, units= "V", title= ""):
    '''
    Takes the unaltered ecg data and plots it with voltage on the y axis and 
    time on the x axis. 

    Parameters
    ----------
    signal_voltage : ndarray
        1D array containing floats of voltages for every sample time.
    signal_time : ndarray 
        A 1D array of floats that has been scaled to seconds instead of index values. 
    units : str, optional
        A string for the units of the y axis. The default is "V".
    title : str, optional
        A string to be the title of the plot. The default is "".

    Returns
    -------
    None.

    '''
    # create figure and plot data 
    plt.figure(1, dpi=200, clear=True)
    plt.plot(signal_time, signal_voltage, c='k', label='Signal')
    
    # annotate plot 
    plt.title(title)
    plt.xlabel('time (s)')
    plt.ylabel(units)
    plt.tight_layout()

# %% Part 3
def plot_events(label_samples, label_symbols, signal_time, signal_voltage):
    '''
    This function plots a dot at the location that was annotated by the expert scorer.  

    Parameters
    ----------
    label_samples : ndarray 
        1D array of integers for the indices that a scorer made a notes at. 
    label_symbols : ndarray
        A 1D array of strings containing the specified labels. 
    signal_time : ndarray
        An array that goes up to the time in seconds of the recording with 
        steps of 1/frequency
    signal_voltage : ndarray
        A 1D array of floats for the voltage values at each timepoint. 

    Returns
    -------
    None.

    '''
    # extract the different types of labels
    annotation_types = np.unique(label_symbols)
    
    # Iterate through different annotation types and plot them on an existing plot of raw data 
    for label in annotation_types:
        # creates boolean mask 
        is_label = label_symbols == label
        plt.scatter(signal_time[label_samples[is_label]], signal_voltage[label_samples[is_label]], label= label)
    plt.legend(loc='lower right')
    
# %% Part 4
def extract_trials(signal_voltage, trial_start_samples, trial_sample_count):
    '''
    This function makes a 2D array with each row containing a 1 second clip 
    from around a heart beat annotation. 

    Parameters
    ----------
    signal_voltage : ndarray
        A 1D array of floats for the voltage values at each timepoint.
    trial_start_samples : ndarray
        A 1D array of integers that provide the starting index for each clip. 
    trial_sample_count : int
        A single value defining the number of values per clip typically the 
        sampling frequency times the time in seconds of the clip. 

    Returns
    -------
    trials : ndarray
        A 2D array containing m samples of n measurements. (m x n array)

    '''
    # create properly sized array 
    trials = np.zeros((len(trial_start_samples), trial_sample_count))
    
    # Iterates through trial start values and cuts out windows assigning each trial to a row
    for trials_index, trial in enumerate(trial_start_samples):
        trials[trials_index, :] = signal_voltage[trial: (trial + trial_sample_count)]
    # returns trial arrays 
    return trials
  

# %% Part 5  
def plot_mean_and_std_trials(signal_voltage, label_samples, label_symbols, trial_duration_seconds, fs, units= "V", title= ""):
    '''
    Wrapper function that compiles all previous functions except for the load data function. 
    Creates a raw data plot, adds even annotation markers, zooms in on example segment, extracts
    trials for all kinds of annotations and then calculates the average signal for each type of annotation.
    Eventually creates a plot of mean signal with standard deviation around it for all annotation types. 

    Parameters
    ----------
    signal_voltage : ndarray
        A 1D array of floats for the voltage values at each timepoint.
    label_samples : ndarray 
        1D array of integers for the indices that a scorer made a notes at. 
    label_symbols : ndarray
        A 1D array of strings containing the specified labels. 
    trial_duration_seconds : int
        Desired time in seconds that each clip will be cut to. 
    fs : int
        The frequency in Hz for sample rate as an integer.
    units : str, optional
        A string for the units of the y axis. The default is "V".
    title : TYPE, optional
        A string to be the title of the plot. The default is "".

    Returns
    -------
    symbols : ndarray 
        List of the different annotation markers that appeared in the data set. 
    trial_time : ndarray 
        A 1D array of length trial_duration_seconds times fs with step size fs. 
    mean_trial_signal : ndarray
        A 2D array with the mean signal clip in the rows for each type of symbol. 

    '''
    time = np.arange(0, len(signal_voltage) / fs, 1/fs)
    adapted_title = 'Raw ' + title
    plot_raw_data(signal_voltage, time, units, adapted_title)
    plt.savefig(f'Unprocessed ECG Signal Subject e0103.png')
    plt.title('Zoomed In Section of Raw ' + title)
    plot_events(label_samples, label_symbols, time, signal_voltage)
    plt.xlim((1909.5, 1912.1))
    plt.tight_layout()
    plt.savefig(f'Zoomed In Window ECG Signal Subject e0103.png')
    annotation_types = np.unique(label_symbols)
    
    annotations_grouped = {}
    
    time_clips = np.arange(0, trial_duration_seconds, 1/fs)
    
    trial_sample_count = trial_duration_seconds * fs
    
    shift_from_annotation = int((trial_sample_count) / 2)
    
    plt.figure(dpi=200)
    for annotation in annotation_types:
        is_annotation = label_symbols == annotation
        annotated_indices = label_samples[is_annotation] - shift_from_annotation
        is_in_range = np.logical_and(annotated_indices > 0, (annotated_indices + trial_sample_count) < len(signal_voltage))
        annotations_grouped[f'{annotation}'] = extract_trials(signal_voltage, annotated_indices[is_in_range], trial_sample_count)
        
        plt.plot(time_clips, annotations_grouped[f'{annotation}'][random.randint(0, len(annotations_grouped[f'{annotation}'])),:], label= annotation)
        plt.title(f'Sample Heart Beats from all Annotation Types\n Subject e0103, Electrode V4')
        plt.xlabel('time (s)')
        plt.ylabel(units)
        plt.grid()
        plt.legend()
        plt.tight_layout()  
    plt.savefig(f'Random Example of ECG Heart Beats Subject e0103.png')
    
    mean_trial_signal = np.zeros((len(annotation_types), (trial_duration_seconds * fs)))
    plt.figure(dpi=200)
    for annotation_index, annotation in enumerate(annotation_types):
        column_avg = np.average(annotations_grouped[f'{annotation}'], axis=0)
        mean_trial_signal[annotation_index, :] = column_avg
        column_std = np.std(annotations_grouped[f'{annotation}'], axis=0)
        plt.plot(time_clips, column_avg, label= f'{annotation}')
        plt.fill_between(time_clips, column_avg-column_std, column_avg+column_std, label= f'STD of {annotation}', alpha=.4)
        plt.title('Mean ' + title)
        plt.xlabel('time (s)')
        plt.ylabel(units)
        plt.grid()
        plt.tight_layout()
        plt.legend()
    
    plt.savefig(f'Mean ECG Heart Beats Subject e0103.png')
    symbols = annotation_types
    trial_time = time_clips
    return symbols, trial_time, mean_trial_signal
       
# %% Part 6
def save_means(symbols, trial_time, mean_trial_signal, out_filename='ecg_means.npz'):
    """
    Function to save data from the pipeline. 

    Parameters
    ----------
    symbols : ndarray
        Array of strings that stores annotation markings.
    trial_time : ndarray
        1D array of time values for each sample in the extracted trials
    mean_trial_signal : ndarray
        Array of mean signal from each annotation type. 
    out_filename : str, optional
        String for the file name to which all data will be saved. The default is 'ecg_means.npz'.

    Returns
    -------
    None.

    """
    # save files
    np.savez(out_filename, symbols= symbols, trial_time= trial_time, mean_trial_signal= mean_trial_signal)
    






























