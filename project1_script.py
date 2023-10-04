# -*- coding: utf-8 -*-
"""
Ben Wilson, Thomas Bausman 
BME3000
Project 1
The script runs through the processing and plotting of an ECG data recording of an individual. The data
has been annotated by a professional and the end goal is to plot the mean heart beat for all annotation 
types. The code below starts by loading data before plotting it unporcessed. Then the annotation events 
are plotted ontop. Next clips of 1 second are extracted around each annotation which are then averaged
and the means are plotted with the standard deviations. Finally all processed data and images are saved.
"""
#%% Part 1
# importing all modules used
import numpy as np 
import matplotlib.pyplot as plt
import project1_module as funct
import random

# loading the data using the load data function from the project module
file = 'ecg_e0103_half1.npz'
signal_voltage, fs, label_samples, label_symbols, subject_id, electrode, units = funct.load_data(file)

# create the scaled time array in seconds with step of 1/fs
time = np.arange(0, len(signal_voltage) / fs, 1/fs)

# %% Part 2
title = f"ECG Raw Signal from subject {subject_id}, electrode {electrode}"

# plot raw signal using function from module
funct.plot_raw_data(signal_voltage, time, title= title, units= units)

# %% Part 3 
# overlay the annotations on the existing plot and zoom in to a specific location
funct.plot_events(label_samples, label_symbols, time, signal_voltage)
plt.xlim((1909.5, 1912.1))
print()

# %% Part 4
# Isolate out the normal annotations, remove edge cases and create a 2D array using function from module
is_normal = label_symbols == 'N'
normal_annotated_indices = label_samples[is_normal] - int(fs/2)
is_normal_in_range = np.logical_and(normal_annotated_indices > 0, (normal_annotated_indices + fs) < len(signal_voltage))
n_count = np.count_nonzero(is_normal_in_range)
normal_trials = funct.extract_trials(signal_voltage, normal_annotated_indices[is_normal_in_range], fs)

# Isolate out the abnormal annotations, remove edge cases and create a 2D array using function from module
is_abnormal = label_symbols == 'V'
v_count = np.count_nonzero(is_abnormal)
abnormal_annotated_indices = label_samples[is_abnormal] - int(fs/2)
is_abnormal_in_range = np.logical_and(abnormal_annotated_indices > 0, (abnormal_annotated_indices + fs) < len(signal_voltage))
abnormal_trials = funct.extract_trials(signal_voltage, abnormal_annotated_indices[is_abnormal_in_range], fs)

# verify that all the trial arrays are the right shape and have been filled with values 
if np.shape(normal_trials) == (n_count, fs):
    print(f'Normal instance array is correct size ({n_count},{fs})')
if np.shape(normal_trials) == (n_count, fs):
    print(f'Abnormal instance array is correct size ({v_count},{fs})')
# the array was initially created and filled with zeros which is why ensuring there are non zero values
if np.count_nonzero(normal_trials) > 0:
    print('The normal array has values')
if np.count_nonzero(abnormal_trials) > 0:
    print('The abnormal array has values')

# create time array for trials
time_clips = np.arange(0, 1, 1/fs)

# create new figure and plot random samples from the trial arrays
plt.figure(3, dpi=200, clear=True)
plt.plot(time_clips, normal_trials[random.randint(0, len(normal_trials)),:], label= 'Sample Normal Heart Beat')
plt.plot(time_clips, abnormal_trials[random.randint(0, len(abnormal_trials)),:], label= 'Sample Abnormal Heart Beat')

# annotate plot
plt.title(f'A Random Sample of Normal and Abnormal Heart Beat from Subject {subject_id}')
plt.xlabel('time (s)')
plt.ylabel(f'{units}')
plt.grid()
plt.legend()
plt.tight_layout()

# %% Part 5 
# define trial duration and call general wrapper function to execute all previous functions except load data 
# and calculate and plot the mean signals. 
trial_duration_seconds = 1
title = f"ECG Signals from Subject {subject_id}, Electrode {electrode}"
symbols, clip_time, average_array = funct.plot_mean_and_std_trials(signal_voltage, label_samples, label_symbols, trial_duration_seconds, fs, units= units, title= title)

# %% Part 6
# save all the new data and reload to ensure the data stored properly. 
out_filename = f'ecg_means_{subject_id}.npz'
funct.save_means(symbols, clip_time, average_array, out_filename)
reloaded_data = np.load(out_filename)

# check that all values are the same
if np.array_equal(reloaded_data['symbols'], symbols):
    print('Symbols: success')
if np.array_equal(reloaded_data['trial_time'],clip_time):
    print('Trial time: success')
if np.array_equal(reloaded_data['mean_trial_signal'], average_array):
    print('Mean signals: success')