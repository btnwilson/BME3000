#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 13:53:20 2023

@author: abbyhoward

project3_module.py
"""
import numpy as np
from matplotlib import pyplot as plt
import scipy

#%% part 1

def remove_peaks(data_dict, fs, threshold):
    no_peak_data = {}
    for key in data_dict:
        signal = data_dict[key]
        
        indicies = np.arange(0,len(signal))
        
        peaks = indicies[signal > threshold]
        
        for index in peaks:
            signal[index] = (signal[index + 1] + signal[index - 1]) / 2
       
        no_peak_data = no_peak_data | {key : signal}
        
    return no_peak_data

def plot_data(signal, fs,title = None, units ='A.U.', label = None):
    time = np.arange(0,len(signal)/fs, 1/fs)
    plt.plot(time, signal, label = label)
    plt.title(title)
    plt.xlabel('time (s)')
    plt.ylabel(units)
    plt.grid()
    plt.legend()
    plt.title(title)
         
  
def filter_data(signal, impulse_response):
    filtered_signal = np.convolve(signal, impulse_response, mode = 'same')
    
    return filtered_signal

def find_beats(signal, fs, flipped = False):
    if flipped == True:
        processed_signal = signal * -1
    else:
        processed_signal = signal
    beats, info = scipy.signal.find_peaks(processed_signal, height = 100)
    time = np.arange(0,len(signal)/fs, 1/fs)
    plt.scatter(time[beats], signal[beats], c = 'r')
    return beats
        
    
def hrv(beat_indicies, fs):
    beat_indicies = np.array(beat_indicies)
    differences = np.diff(beat_indicies/fs)
    hrv = np.std(differences)
    return differences, hrv

def plot_frequency_bands(signal, fs, low_fc_range, high_fc_range, title = None, units = 'A.U.'):
    fft = scipy.fft.rfft(signal)
    freq = scipy.fft.rfftfreq(len(signal), fs)
    
    low_fc_mask = (freq >= low_fc_range[0]) & (freq <= low_fc_range[1])
    high_fc_mask = (freq >= high_fc_range[0]) & (freq <= high_fc_range[1])
    
    low_fc_fft = fft[low_fc_mask]
    low_fc = np.arange(low_fc_range[0], low_fc_range[1], ((low_fc_range[1] - low_fc_range[0]) / len(low_fc_fft)))
    
    high_fc_fft = fft[high_fc_mask]
    high_fc = np.arange(high_fc_range[0], high_fc_range[1], ((high_fc_range[1] - high_fc_range[0]) / len(high_fc_fft)))
   
    fft_power = ((np.abs(fft)) ** 2)
    
    plt.plot(freq, fft_power, c = 'gray', zorder = 0 )
    plt.fill_between(low_fc, np.abs(fft_power[low_fc_mask]), label = 'low frequecy band')
    plt.fill_between(high_fc, np.abs(fft_power[high_fc_mask]), label = 'high frequency band')
    
    plt.title(title)
    plt.ylabel(units)
    plt.xlabel('frequencies')
    plt.xlim(0,high_fc_range[1])
    plt.legend()
    plt.grid()
    mean_low_fc = np.mean(np.abs(low_fc_fft))
    mean_high_fc = np.mean(np.abs(high_fc_fft))
    ratio = mean_low_fc / mean_high_fc
    
    return ratio
    
    














    













