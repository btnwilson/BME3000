#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 13:53:20 2023

@authors: Abby Howard and Ben Wilson

project3_module.py

This module contains 6 functions utilized in project3_script. The first removes large artifact spikes, replacing the values
by interpolation using the values from before and after the spike. The second plots a time domain signal and labels the plot.
Function three convolves a signal and impulse response to return a filtered signal. While the fourth determines where heart beats
are using Scipy's find_peaks function and a threshold value to return the indicies where beats occur. The fifth function determines
the heart rate variability using the beat indicies and finding the inter-beat intervals to returning the standard deviation. The
final function completes the fourier transform of a signal and masks low and high-frequency bands, then plots the signal with the
bands labeled. It also determines the mean power of the bands and returns the ratio of the low to high frequency power.
"""
import numpy as np
from matplotlib import pyplot as plt
import scipy

#%% part 1
def remove_spikes(data_dict, fs, threshold):
    '''
    This function removes spikes that are above a given threshold that
    are likely outlires in a set of data. Works with a dictionary of data 
    sets. 
    

    Parameters
    ----------
    data_dict : dictionary of size n where n is the number of signal recordings.
        Is a dictionary of signals that are some kind of number sequency, either a list or array. 
    fs : integer
        The sampling rate for the signals being cleaned
    threshold : float
        the value above which a data point is removed and replaced by an interpolated value.

    Returns
    -------
    no_spike_data : dictionary of size n where n is the number of signal recordings, same size as input dictionary
        Dictionary of signals with any spikes removed.

    '''
    # create empty dictionary 
    no_spike_data = {}
    
    # loop through signals in dictionary entered to function
    for key in data_dict:
        # extract the individual signal and create an indices array
        signal = data_dict[key]
        indices = np.arange(0,len(signal))
        
        # create a boolean mask of indices where values exceed threshold
        is_spike = indices[signal > threshold]
        
        # loop through spikes and remove and interpolate for the value
        for index in is_spike:
            signal[index] = (signal[index + 1] + signal[index - 1]) / 2
        
        # place cleaned signal in dictionary to be returned 
        no_spike_data = no_spike_data | {key : signal}
        
    return no_spike_data

def plot_data(signal, fs, title = None, units ='A.U.', label = None):
    '''
    Simple plotter function that takes a signal and constructs a time array 
    for it then plots the signal in the time domain and annotates the plots 

    Parameters
    ----------
    signal : 1D array of floats of size (n,) where n is the number of samples 
        The signal to be plotted.
    fs : integer
        The sampling frequency of the signal. 
    title : string, optional
        The title of the plot that will be created of the input signal. The default is None.
    units : string, optional
        The y axis label containg the units of the y axis. The default is 'A.U.'.
    label : string, optional
        The label that will be displayed in the legend for the signal if needed. The default is None.

    Returns
    -------
    None.

    '''
    # create time vector 
    time = np.arange(0,len(signal)/fs, 1/fs)
    
    # plot the signal
    plt.plot(time, signal, label = label)
    
    # annotate the plot
    plt.title(title)
    plt.xlabel('time (s)')
    plt.ylabel(units)
    plt.grid()
    plt.title(title)
         
    
#%% Part 2
def filter_data(signal, impulse_response):
    """
    This function filters the signal provided using the impulse 
    response of a filter provided using convolution. 

    Parameters
    ----------
    signal : 1D array of floats size (n,) where n is the number of samples in the signal
        Signal to be filtered. 
    impulse_response : 1D array of floats size (n,) where n is the number of samples in the impulse response
        The impulse response of the filter to be used on the signal. 

    Returns
    -------
    filtered_signal : array of floats size (n,) where n is the number of samples in the signal
        The filtered signal which has been convolved with the impusle response the same size as input signal.

    """
    # filter the signal via convolution of impulse response
    filtered_signal = np.convolve(signal, impulse_response, mode = 'same')
    
    return filtered_signal

#%% part 3
def find_beats(signal, fs, threshold, flipped = False, plot = False):
    """
    This function detects all peak values that exceed a certain threshold.
    Used to detect heartbeats or other events that coincide with a spike in
    a signal. This function returns the indices for the original signal of the peaks.
    This function can also create a time array and plots the peaks as a scatter plot. 
    Parameters
    ----------
    signal : 1D array of floats size (n,) where n is the number of samples in the signal
        Signal have the peaks detected from.
    fs : integer
        The sampling frequency of the signal.
    threshold : float
        The value above which a peak will be detected. If a local maximum does not 
        exceed this value it will not be flagged. 
    flipped : Boolean, optional
        If true the signal will be inverted before processing so effectively it will
        detect trougths. The default is False.
    plot : Boolean, optioal
        If true the detected peaks will be plotted by indexing the signal. The default is False

    Returns
    -------
    beats : 1D array of floats with a shape (n,) where n is the number of peaks detected
        An array containing the indices where peak values were detected. 

    """
    # flip the signal if necessary 
    if flipped == True:
        processed_signal = signal * -1
    else:
        processed_signal = signal
    
    # using a scipy function find all of the peak above a given threshold
    beats, info = scipy.signal.find_peaks(processed_signal, height = threshold)
    
    # plot the beats if necessary
    if plot == True:
        # create a time vector and plot the events on the currently opened axis object
        time = np.arange(0,len(signal)/fs, 1/fs)
        plt.scatter(time[beats], signal[beats], c = 'r')
        
    # return beat indices
    return beats
        
#%% part 4
def hrv(beat_indices, fs):
    '''
    This function calculates the heart rate variability given an
    array of the beat indices and the sampling frequency. Heart
    rate variabilty is the standard deviation of Inter-beat interval.

    Parameters
    ----------
    beat_indices : 1D array of floats with size (n, ) where n is the number of beats
        An array containing the indices of each beat or spike
    fs : integer
        The sampling frequency of the signal.

    Returns
    -------
    differences : 1D array of floats of size (n, ) where n is the number of beats minus 1
        An array of the time differences between each beat or spike. 
    hrv : float
        The standard deviation of the differences array, used as a metric for regularity. 

    '''
    # convert to a numpy array 
    beat_indices = np.array(beat_indices)
    # convert indices to time and find the difference between each time of a beat
    differences = np.diff(beat_indices/fs)
    # calculate HRV
    hrv = np.std(differences)
    # return both IBI and HRV
    return differences, hrv

#%% part 5
def plot_frequency_bands(signal, fs, low_fc_range, high_fc_range, title = None, units = 'A.U.'):
    '''
    This function computes the FFT of the input signal and plots the power in the frequency domain. 
    It also isolates a low and high frequency band as specified by the inputs
    and displays those regions on the plot. Then the mean power of the band is calculated
    and the ratio of low frequency power to high frequency power is computed. This ratio
    is used to approximate sympathetic nervous system activity based on the inter-beat interval
    signal from an ECG recording. 

    Parameters
    ----------
    signal : 1D array of floats size (n,) where n is the number of samples in the signal
        Signal to compute FFT and isolate frequency bands on.
    fs : integer
        The sampling frequency of the signal.
    low_fc_range : 1D list or array of floats size 2 or shape (2,)
        The bounds of the low frequency band to be extracted. 
    high_fc_range : 1D list or array of floats size 2 or shape (2,)
        The bounds of the high frequency band to be extracted.
    title : string, optional
        The title of the plot that will be created of the input signal in the frequency domain. 
        The default is None.
    units : string, optional
        The y axis label containg the units of the y axis. The default is 'A.U.'.

    Returns
    -------
    ratio : float
        The low to high frequency ratio of the mean power within the frequency bands. 

    '''
    # compute the fft of the signal and the corresponding frequencies for the x axis
    fft = scipy.fft.rfft(signal)
    freq = scipy.fft.rfftfreq(len(signal), fs)
    
    # convert fft output to power
    fft_power = np.abs(fft** 2)
    
    # create boolean masks for the frequency bands 
    is_low_fc_mask = (freq >= low_fc_range[0]) & (freq <= low_fc_range[1])
    is_high_fc_mask = (freq >= high_fc_range[0]) & (freq <= high_fc_range[1])
    
    # use boolean mask to isolate frequency bands
    low_fc_fft = fft_power[is_low_fc_mask]
    low_fc = freq[is_low_fc_mask]
    high_fc_fft = fft_power[is_high_fc_mask]
    high_fc = freq[is_high_fc_mask]
    
    
    # plot the fft power of the signal
    plt.plot(freq, fft_power, c = 'gray', zorder = 0 )
    
    # plot the frequency bands 
    plt.fill_between(low_fc, np.abs(low_fc_fft), label = 'low frequecy band')
    plt.fill_between(high_fc, np.abs(high_fc_fft), label = 'high frequency band')
    
    # annotate and format plot
    plt.title(title)
    plt.ylabel(units)
    plt.xlabel('frequency (Hz)')
    plt.xlim(0,high_fc_range[1])
    plt.legend()
    plt.grid()
    
    # compute the mean powers of the frequency bands
    mean_low_fc = np.mean(np.abs(low_fc_fft))
    mean_high_fc = np.mean(np.abs(high_fc_fft))
    
    # compute and return the low freq/ high freq ratio
    ratio = mean_low_fc / mean_high_fc
    
    return ratio
    









    













