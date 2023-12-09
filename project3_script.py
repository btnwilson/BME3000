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
fs = 500

path = 'recorded_data'
files = os.listdir(path)

data = {}
for file in files:
    if '.txt' in file:
        dictionary = {file.strip('.txt'): np.loadtxt(f'{path}/{file}')}
        data = data | dictionary
        

no_peak_data = p3m.remove_peaks(data, fs, 1000)

relaxing_sitting = no_peak_data['relaxing_sitting'] * (5/1023)
relaxing_activity = no_peak_data['relaxing_activity'] * (5/1023)
mentally_stressful = (no_peak_data['mentally_stressful'][:(300 * fs)]) * (5/1023)
physically_stressful = no_peak_data['physically_stressful'] * (5/1023)


plt.figure(1, clear = True)
plt.title('Raw Data')

plt.subplot(3,2,1)
p3m.plot_data(relaxing_sitting,fs, 'Relaxing Sitting Signal', 'Volts (V)')
plt.xlim(165,170)

plt.subplot(3,2,2)
p3m.plot_data(relaxing_activity, fs, 'Relaxing Activity Signal', 'Volts (V)')
plt.xlim(255,260)

plt.subplot(3,2,3)
p3m.plot_data(mentally_stressful, fs, 'Mentally Stressful Signal','Volts (V)')
plt.xlim(125,130)

plt.subplot(3,2,4)
p3m.plot_data(physically_stressful, fs, 'Physically Stressful Signal','Volts (V)')
plt.xlim(145,150)



plt.subplot(3,1,3)
concatenated_signal = np.concatenate([relaxing_sitting, relaxing_activity, mentally_stressful, physically_stressful])

plt.title('Concatenated Signal')
p3m.plot_data(concatenated_signal, fs, 'Concatenated Signal', 'Volts (V)')
plt.tight_layout()
plt.savefig('raw_data')

#%% part 2
# define filter perameters (band-stop filter)

# low cutoff
fc_low = 0.67
#high cutoff
fc_high = 40

numtaps = 501

#band stop
impulse_response = signal.firwin(numtaps, [fc_low,fc_high], fs = fs, pass_zero = False, window = 'hann')

plt.figure(2, clear = True)

plt.subplot(1,2,1)
p3m.plot_data(impulse_response, fs, 'Impulse Response')

plt.subplot(1,2,2)
fft_response = fft.rfft(impulse_response)
f_filter = fft.rfftfreq(len(impulse_response), 1/fs)
plt.plot(f_filter, np.abs(fft_response))
plt.ylabel('|X(f)| A.U.')
plt.title('Frequency Domain of Filter')
plt.xlabel('frequency (Hz)')
plt.grid()
plt.tight_layout()

plt.savefig('filter.png')

plt.figure(3, clear = True)
plt.subplot(2,1,1)
p3m.plot_data(relaxing_sitting, fs, label= 'original signal')
filt_relaxing_sitting = p3m.filter_data(relaxing_sitting, impulse_response)
p3m.plot_data(filt_relaxing_sitting, fs, title = 'Relaxing Sitting', label = 'filtered signal', units = 'Volts (V)')
plt.xlim(165,170)

plt.grid()

plt.subplot(2,1,2)
p3m.plot_data(mentally_stressful, fs,  label = 'original signal' )
filt_mentally_stressful = p3m.filter_data(mentally_stressful, impulse_response)
p3m.plot_data(filt_mentally_stressful, fs, title = 'Mentally Stressful',label = 'filtered signal', units = 'Volts (V)')
plt.xlim(125,130)
plt.grid()

plt.tight_layout()
plt.savefig('filtered_signal')


plt.figure(4, clear = True)
plt.title('Filtered Signals With Beats')

plt.subplot(2,2,1)
p3m.plot_data(filt_relaxing_sitting, fs, 'Filtered Relaxing Sitting Signal', 'Volts (V)')

plt.subplot(2,2,2)
filt_relaxing_activity = p3m.filter_data(relaxing_activity, impulse_response)
p3m.plot_data(filt_relaxing_activity, fs, 'Filtered Relaxing Activity Signal', 'Volts (V)')

plt.subplot(2,2,3)
p3m.plot_data(filt_mentally_stressful, fs, 'Filtered Mentally Stressful Signal', 'Volts (V)')

plt.subplot(2,2,4)
filt_physically_stressful = p3m.filter_data(physically_stressful,impulse_response)
p3m.plot_data(filt_physically_stressful, fs, 'Filtered Physically Stressful Signal', 'Volts (V)')

plt.tight_layout()

#%% part 3

plt.figure(4)

plt.subplot(2,2,1)
relaxing_sitting_beats = p3m.find_beats(filt_relaxing_sitting, fs, 0.5, flipped = True)

plt.subplot(2,2,2)
relaxing_activity_beats = p3m.find_beats(filt_relaxing_activity, fs, 0.5, flipped = True)

plt.subplot(2,2,3)
mentally_stressful_beats = p3m.find_beats(filt_mentally_stressful, fs, 0.5,  flipped = True)

plt.subplot(2,2,4)
physically_stressful_beats = p3m.find_beats(filt_physically_stressful, fs,0.5,  flipped = True)

#%% part 4
rs_ibi, rs_hrv = p3m.hrv(relaxing_sitting_beats, fs)

ra_ibi, ra_hrv = p3m.hrv(relaxing_activity_beats, fs)

ms_ibi, ms_hrv = p3m.hrv(mentally_stressful_beats, fs)

ps_ibi, ps_hrv = p3m.hrv(physically_stressful_beats, fs)

plt.figure(5, clear = True)
plt.title('Heart Rate Veriabilities')

plt.bar(['relaxing sitting', 'relaxing activity', 'mentally stressful', 'physically stressful'],[rs_hrv, ra_hrv, ms_hrv, ps_hrv])
plt.xlabel('Activity Type')
plt.ylabel('Time (s)')
plt.grid()
plt.tight_layout()


rs_time = np.arange(0,len(relaxing_sitting) / fs, 0.1)
rs_interp = np.interp(rs_time, relaxing_sitting_beats[1:] / fs, rs_ibi)

ra_time = np.arange(0,len(relaxing_activity) / fs, 0.1)
ra_interp = np.interp(ra_time, relaxing_activity_beats[1:] / fs, ra_ibi)

ms_time = np.arange(0,len(mentally_stressful) / fs, 0.1)
ms_interp = np.interp(ms_time, mentally_stressful_beats[1:] / fs, ms_ibi)

ps_time = np.arange(0,len(physically_stressful) / fs, 0.1)
ps_interp = np.interp(ps_time, physically_stressful_beats[1:] / fs, ps_ibi)

plt.figure(6, clear = True)

plt.subplot(2,2,1)
rs_ratio = p3m.plot_frequency_bands(rs_interp, 0.1, [0.04,0.15], [0.15,0.4], title = 'Relaxing Sitting')
plt.ylim(0,3000)
plt.subplot(2,2,2)
ra_ratio = p3m.plot_frequency_bands(ra_interp, 0.1, [0.04,0.15], [0.15,0.4], title = 'Relaxing Activity')
plt.ylim(0,8000)
plt.subplot(2,2,3)
ms_ratio = p3m.plot_frequency_bands(ms_interp, 0.1, [0.04,0.15], [0.15,0.4], title = 'Mentally Stressful')
plt.ylim(0,3000)
plt.subplot(2,2,4)
ps_ratio = p3m.plot_frequency_bands(ps_interp, 0.1, [0.04,0.15], [0.15,0.4], title = 'Physically Stressful')
plt.ylim(0,3000)
plt.tight_layout()

plt.figure(7, clear = True)
plt.title('LF/HF Ratios')
plt.bar(['relaxing sitting','relaxing activity','mentally stressful','physically stressful'], [rs_ratio, ra_ratio, ms_ratio, ps_ratio])
plt.xlabel('Activity Type')
plt.ylabel('A.U.')
plt.grid()
plt.tight_layout()





