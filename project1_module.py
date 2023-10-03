# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 08:44:53 2023

@author: bentn
"""

import numpy as np
import matplotlib.pyplot as plt
import random
def load_data(file_name):
    files = np.load(file_name)
    print('Data in file:')
    for file in files.files:
        print(file + ',')
    ecg_voltage = files['ecg_voltage']
    electrode = files['electrode']
    frequency = files['fs']
    labeled_samples_indicies = files['label_samples']
    labeled_symbols = files['label_symbols']
    subject_id = files['subject_id']
    units = files['units']
    
    return ecg_voltage, frequency, labeled_samples_indicies, labeled_symbols, subject_id, electrode, units
    
   
def plot_raw_data(signal_voltage, signal_time, units= "V", title= ""):
    plt.figure(1, dpi=200, clear=True)
    plt.plot(signal_time, signal_voltage, c='k')
    plt.title(title)
    plt.xlabel('time (s)')
    plt.ylabel(units)
    plt.tight_layout()
    
def plot_events(label_samples, label_symbols, signal_time, signal_voltage):
    annotation_types = np.unique(label_symbols)
    for label in annotation_types:
        is_label = label_symbols == label
        plt.scatter(signal_time[label_samples[is_label]], signal_voltage[label_samples[is_label]], label= label)
    plt.legend()
    plt.xlim((1909.5, 1912.1))
        
def extract_trials(signal_voltage, trial_start_samples, trial_sample_count):
    trials = np.zeros((len(trial_start_samples), trial_sample_count))
    for trials_index, trial in enumerate(trial_start_samples):
        trials[trials_index, :] = signal_voltage[trial: (trial + trial_sample_count)]
    return trials
    
def plot_mean_and_std_trials(signal_voltage, label_samples, label_symbols, trial_duration_seconds, fs, units= "V", title= ""):
    time = np.arange(0, len(signal_voltage) / fs, 1/fs)
    plot_raw_data(signal_voltage, time)
    plot_events(label_samples, label_symbols, time, signal_voltage)
    
    cleaned_labeled_samples = label_samples[1:-1]
    
    annotation_types = np.unique(label_symbols)
    
    annotations_grouped = {}
    
    time_clips = np.arange(0, trial_duration_seconds, 1/fs)
    
    shift_from_annotation = int((trial_duration_seconds * fs) / 2)
    for annotation in annotation_types:
        is_annotation = label_symbols[1:-1] == annotation
        annotated_indicies = cleaned_labeled_samples[is_annotation]
        annotated_trial_start_indicies = annotated_indicies - shift_from_annotation
        annotations_grouped[f'{annotation}'] = extract_trials(signal_voltage, annotated_trial_start_indicies, fs)
        
        plt.figure(dpi=200)
        plt.plot(time_clips, annotations_grouped[f'{annotation}'][random.randint(0, len(annotations_grouped[f'{annotation}'])),:])
        plt.title(f'Sample Heart Beat labeled {annotation}')
        plt.xlabel('time (s)')
        plt.ylabel('V')
        plt.grid()
        plt.tight_layout()  
    
    mean_trial_signal = np.zeros((len(annotation_types), (trial_duration_seconds * fs)))
    plt.figure(dpi=200)
    for annotation_index, annotation in enumerate(annotation_types):
        column_avg = np.average(annotations_grouped[f'{annotation}'], axis=0)
        mean_trial_signal[annotation_index, :] = column_avg
        column_std = np.std(annotations_grouped[f'{annotation}'], axis=0)
        plt.plot(time_clips, column_avg, label= f'{annotation}')
        plt.fill_between(time_clips, column_avg-column_std, column_avg+column_std, label= f'STD of {annotation}', alpha=.4)
        plt.title(title)
        plt.xlabel('time (s)')
        plt.ylabel(units)
        plt.grid()
        plt.tight_layout()
        plt.legend()
    
    symbols = annotation_types
    trial_time = time_clips
    return symbols, trial_time, mean_trial_signal
       
    
    






























