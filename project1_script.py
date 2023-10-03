# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 09:10:51 2023

@author: bentn
"""
import numpy as np 
import matplotlib.pyplot as plt
import project1_module as funct
import random
file = 'ecg_e0103_half1.npz'
signal_voltage, fs, label_samples, label_symbols, subject_id, electrode, units = funct.load_data(file)
time = np.arange(0, len(signal_voltage) / fs, 1/fs)
funct.plot_raw_data(signal_voltage, time, title="ECG Raw Signal")
funct.plot_events(label_samples, label_symbols, time, signal_voltage)

cleaned_labeled_samples = label_samples[1:-1]

is_normal = label_symbols[1:-1] == 'N'
normal_indicies = cleaned_labeled_samples[is_normal]
normal_trial_start_indicies = normal_indicies - int((fs / 2))
normal_trials = funct.extract_trials(signal_voltage, normal_trial_start_indicies, fs)

is_abnormal = label_symbols[1:-1] == 'V'
abnormal_indicies = cleaned_labeled_samples[is_abnormal]
abnormal_trial_start_indicies = abnormal_indicies - int((fs / 2))
abnormal_trials = funct.extract_trials(signal_voltage, abnormal_trial_start_indicies, fs)

# %%
time_clips = np.arange(0, 1, 1/fs)

plt.figure(3, dpi=200, clear=True)
plt.plot(time_clips, normal_trials[random.randint(0, len(normal_trials)),:])
plt.title('Sample Normal Heart Beat')
plt.xlabel('time (s)')
plt.ylabel('V')
plt.grid()
plt.tight_layout()

plt.figure(4, dpi=200, clear=True)
plt.plot(time_clips, abnormal_trials[random.randint(0, len(abnormal_trials)),:])
plt.title('Sample Abnormal Heart Beat')
plt.xlabel('time (s)')
plt.ylabel('V')
plt.grid()
plt.tight_layout()
#%%
annotation_types = np.unique(label_symbols)
 
annotations_grouped = {}
 
for annotation in annotation_types:
    is_annotation = label_symbols[1:-1] == annotation
    annotated_indicies = cleaned_labeled_samples[is_annotation]
    annotated_trial_start_indicies = annotated_indicies - int((fs / 2))
    annotations_grouped[f'{annotation}'] = funct.extract_trials(signal_voltage, annotated_trial_start_indicies, fs)
column_avg = np.average(annotations_grouped[f'{annotation}'], axis=0)
column_std = np.std(annotations_grouped[f'{annotation}'], axis=0)
plt.plot(time_clips, column_avg, label= f'{annotation}')
plt.fill_between(time, column_avg-column_std, column_avg+column_std, label= f'STD of {annotation}')

plt.xlabel('time (s)')
plt.ylabel(units)
plt.grid()
plt.tight_layout()
# %%
trial_duration_seconds = 1
symbols, clip_time, average_array = funct.plot_mean_and_std_trials(signal_voltage, label_samples, label_symbols, trial_duration_seconds, fs)
"""
total_time_seconds = (len(data['ecg_voltage'])/ data['fs'])
sample_spacing = 1 / frequency 
total_time_seconds = (len(data['ecg_voltage'])/ data['fs'])
time = np.arange(0, total_time_seconds, frequency)
plt.figure(1, figsize=(100,10))
plt.plot(time, data['ecg_voltage'])
"""