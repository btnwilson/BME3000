# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 10:55:48 2023

@author: abbyhoward

project3_script
"""
#%% import packages
import numpy as np
from matplotlib import pyplot as plt
from scipy import fft
from scipy import signal
import os
import project3_module as p3m

#%% part 1

# define the sampling rate
fs = 500

# specify file location where data is stored and gather list of file names
path = 'recorded_data'
files = os.listdir(path)

# make an empty dictionary for data
data = {}

# load data in from every data file in the specified folder and place it in the data dictionary.
for file in files:
    if '.txt' in file:
        dictionary = {file.strip('.txt'): np.loadtxt(f'{path}/{file}')}
        data = data | dictionary
        
# remove the spikes seen in the data collected 
no_peak_data = p3m.remove_peaks(data, fs, 1000)

# Create reference variables for each individual recording and convert the y values to volts
relaxing_sitting = no_peak_data['relaxing_sitting'] * (5/1023)
relaxing_activity = no_peak_data['relaxing_activity'] * (5/1023)
mentally_stressful = (no_peak_data['mentally_stressful'][:(300 * fs)]) * (5/1023)
physically_stressful = no_peak_data['physically_stressful'] * (5/1023)

# create a figure for raw data 
plt.figure(1, clear = True)
plt.title('Raw Data')

# plot relaxing data and annotate within function plus zoom in on 5 second segment
plt.subplot(3,2,1)
p3m.plot_data(relaxing_sitting,fs, 'Relaxing Sitting Signal', 'Volts (V)')
plt.xlim(165,170)

# plot relaxing activity data and annotate within function plus zoom in on 5 second segment
plt.subplot(3,2,2)
p3m.plot_data(relaxing_activity, fs, 'Relaxing Activity Signal', 'Volts (V)')
plt.xlim(255,260)

# plot mentally stressful data and annotate within function plus zoom in on 5 second segment
plt.subplot(3,2,3)
p3m.plot_data(mentally_stressful, fs, 'Mentally Stressful Signal','Volts (V)')
plt.xlim(125,130)

# plot physically stressful data and annotate within function plus zoom in on 5 second segment
plt.subplot(3,2,4)
p3m.plot_data(physically_stressful, fs, 'Physically Stressful Signal','Volts (V)')
plt.xlim(145,150)


# plot/ create concatenated signal and annotate within function
plt.subplot(3,1,3)
concatenated_signal = np.concatenate([relaxing_sitting, relaxing_activity, mentally_stressful, physically_stressful])

# annotate concatenated signal plot
plt.title('Concatenated Signal')
p3m.plot_data(concatenated_signal, fs, 'Concatenated Signal', 'Volts (V)')
plt.tight_layout()
plt.savefig('raw_data')

#%% part 2
# define filter parameters (band-stop filter)

# low cutoff
fc_low = 0.67
#high cutoff
fc_high = 40

# parameter for number of coeffs in the filter
numtaps = 501

# create band stop filter
impulse_response = signal.firwin(numtaps, [fc_low,fc_high], fs = fs, pass_zero = False, window = 'hann')

# create new figure
plt.figure(2, clear = True)

# plot the impulse response in a subplot and annotate using function
plt.subplot(1,2,1)
p3m.plot_data(impulse_response, fs, 'Impulse Response')

# plot/ compute the FFT of the inpulse response 
plt.subplot(1,2,2)
fft_response = fft.rfft(impulse_response)
f_filter = fft.rfftfreq(len(impulse_response), 1/fs)
plt.plot(f_filter, np.abs(fft_response))

# annotate plot
plt.ylabel('|X(f)| A.U.')
plt.title('Frequency Domain of Filter')
plt.xlabel('frequency (Hz)')
plt.grid()
plt.tight_layout()
plt.savefig('filter.png')

# create new figure 
plt.figure(3, clear = True)

# filter the baseline recording and plot it using functions
plt.subplot(2,1,1)
p3m.plot_data(relaxing_sitting, fs, label= 'original signal')
filt_relaxing_sitting = p3m.filter_data(relaxing_sitting, impulse_response)
p3m.plot_data(filt_relaxing_sitting, fs, title = 'Relaxing Sitting', label = 'filtered signal', units = 'Volts (V)')
# zoom into 5 second segment
plt.xlim(165,170)
plt.grid()

# filter the mentally stressful recording and plot it using functions
plt.subplot(2,1,2)
p3m.plot_data(mentally_stressful, fs,  label = 'original signal' )
filt_mentally_stressful = p3m.filter_data(mentally_stressful, impulse_response)
p3m.plot_data(filt_mentally_stressful, fs, title = 'Mentally Stressful',label = 'filtered signal', units = 'Volts (V)')

# Zoom in and format plot then save the figure
plt.xlim(125,130)
plt.grid()
plt.tight_layout()
plt.savefig('filtered_signal')

# create a new figure to show the beat identification
plot.figure(4, clear = True)
plt.title('Filtered Signals With Beats')

# plot the filtered baseline recording
plt.subplot(2,2,1)
p3m.plot_data(filt_relaxing_sitting, fs, 'Filtered Relaxing Sitting Signal', 'Volts (V)')

# filter and plot the relaxing activity recording
plt.subplot(2,2,2)
filt_relaxing_activity = p3m.filter_data(relaxing_activity, impulse_response)
p3m.plot_data(filt_relaxing_activity, fs, 'Filtered Relaxing Activity Signal', 'Volts (V)')

# plot the filtered mentally stressful activity recording
plt.subplot(2,2,3)
p3m.plot_data(filt_mentally_stressful, fs, 'Filtered Mentally Stressful Signal', 'Volts (V)')

# filter and plot the physically stressful activity recording
plt.subplot(2,2,4)
filt_physically_stressful = p3m.filter_data(physically_stressful,impulse_response)
p3m.plot_data(filt_physically_stressful, fs, 'Filtered Physically Stressful Signal', 'Volts (V)')

# fit plots to figure neatly 
plt.tight_layout()

#%% part 3
# re-reference figure
plt.figure(4)

# Re-reference axis object and use function to find the beats in baseline and plot them on the referenced subplot
plt.subplot(2,2,1)
relaxing_sitting_beats = p3m.find_beats(filt_relaxing_sitting, fs, 0.5, flipped = True)

# Re-reference axis object and use function to find the beats in relaxing activity and plot them on the referenced subplot
plt.subplot(2,2,2)
relaxing_activity_beats = p3m.find_beats(filt_relaxing_activity, fs, 0.5, flipped = True)

# Re-reference axis object and use function to find the beats in mentally stressful and plot them on the referenced subplot
plt.subplot(2,2,3)
mentally_stressful_beats = p3m.find_beats(filt_mentally_stressful, fs, 0.5,  flipped = True)

# re-reference axis object and use function to find the beats in physically stressful and plot them on the referenced subplot
plt.subplot(2,2,4)
physically_stressful_beats = p3m.find_beats(filt_physically_stressful, fs,0.5,  flipped = True)

#%% part 4
# compute IBI and HRV using function for baseline
rs_ibi, rs_hrv = p3m.hrv(relaxing_sitting_beats, fs)

# compute IBI and HRV using function for relaxing activity
ra_ibi, ra_hrv = p3m.hrv(relaxing_activity_beats, fs)

# compute IBI and HRV using function for mentally stressful
ms_ibi, ms_hrv = p3m.hrv(mentally_stressful_beats, fs)

# compute IBI and HRV using function for physically stressful
ps_ibi, ps_hrv = p3m.hrv(physically_stressful_beats, fs)

# create a new figure for HRV bar chart 
plt.figure(5, clear = True)
plt.title('Heart Rate Veriabilities')

# plot HRV bar chart 
plt.bar(['relaxing sitting', 'relaxing activity', 'mentally stressful', 'physically stressful'],[rs_hrv, ra_hrv, ms_hrv, ps_hrv])

# annotate and format plot 
plt.xlabel('Activity Type')
plt.ylabel('Time (s)')
plt.grid()
plt.tight_layout()

# create time array and interpolate for IBI values at all time points for baseline 
rs_time = np.arange(0,len(relaxing_sitting) / fs, 0.1)
rs_interp = np.interp(rs_time, relaxing_sitting_beats[1:] / fs, rs_ibi)

# create time array and interpolate for IBI values at all time points for relaxing activity
ra_time = np.arange(0,len(relaxing_activity) / fs, 0.1)
ra_interp = np.interp(ra_time, relaxing_activity_beats[1:] / fs, ra_ibi)

# create time array and interpolate for IBI values at all time points for mentally stressful
ms_time = np.arange(0,len(mentally_stressful) / fs, 0.1)
ms_interp = np.interp(ms_time, mentally_stressful_beats[1:] / fs, ms_ibi)

# create time array and interpolate for IBI values at all time points for physically stressful
ps_time = np.arange(0,len(physically_stressful) / fs, 0.1)
ps_interp = np.interp(ps_time, physically_stressful_beats[1:] / fs, ps_ibi)

# Create new figure for Frequency spectrums 
plt.figure(6, clear = True)

# compute frequency domain values of baseline and return frequency ratio using function
plt.subplot(2,2,1)
rs_ratio = p3m.plot_frequency_bands(rs_interp, 0.1, [0.04,0.15], [0.15,0.4], title = 'Relaxing Sitting')
# zoom into appropriate y range 
plt.ylim(0,3000)

# compute frequency domain values of resting activity and return frequency ratio using function
plt.subplot(2,2,2)
ra_ratio = p3m.plot_frequency_bands(ra_interp, 0.1, [0.04,0.15], [0.15,0.4], title = 'Relaxing Activity')
# zoom into appropriate y range
plt.ylim(0,8000)

# compute frequency domain values of mentally stressful and return frequency ratio using function
plt.subplot(2,2,3)
ms_ratio = p3m.plot_frequency_bands(ms_interp, 0.1, [0.04,0.15], [0.15,0.4], title = 'Mentally Stressful')
# zoom into appropriate y range
plt.ylim(0,3000)

# compute frequency domain values of physically stressful and return frequency ratio using function
plt.subplot(2,2,4)
ps_ratio = p3m.plot_frequency_bands(ps_interp, 0.1, [0.04,0.15], [0.15,0.4], title = 'Physically Stressful')
# zoom into appropriate y range and adjust plot spacings 
plt.ylim(0,3000)
plt.tight_layout()

# create figue for plot and annotate a bar chart for frequency ratios
plt.figure(7, clear = True)
plt.title('LF/HF Ratios')
plt.bar(['relaxing sitting','relaxing activity','mentally stressful','physically stressful'], [rs_ratio, ra_ratio, ms_ratio, ps_ratio])
plt.xlabel('Activity Type')
plt.ylabel('A.U.')
plt.grid()
plt.tight_layout()





