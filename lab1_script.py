# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 11:04:10 2023

@author: bentn
"""

import numpy as np
import matplotlib.pyplot as plt
from lab1_module import plot_histogram
# %%
data_path = 'maternal+health+risk\Maternal Health Risk Data Set.csv'
data = np.loadtxt(data_path, dtype='object', delimiter=',')
column_headers = data[0,:]
column_headers[0] = "Age"
data = np.delete(data, 0, 0)
# %%
patient_ages = data[:,0].astype(int)
patient_risk = data[:,6].astype(str)
for example_index in range(len(data)):
    print(f'Example {example_index}: {column_headers[0]} = {data[example_index,0]}, {column_headers[6]} = {data[example_index,6]}')
high_risk_patient_mask = patient_risk == 'high risk'
mid_risk_patient_mask = patient_risk == 'mid risk'
low_risk_patient_mask = patient_risk == 'low risk'
high_risk_count = 0
for example_index, is_high_risk in enumerate(high_risk_patient_mask):
    if is_high_risk == True:
        print(f'High Risk {high_risk_count}: {column_headers[0]} = {patient_ages[example_index]}')
        high_risk_count += 1
        
        
high_risk = patient_ages[high_risk_patient_mask]
mid_risk = patient_ages[mid_risk_patient_mask]
low_risk = patient_ages[low_risk_patient_mask]
bins = np.arange(9.5,68.5, 1)
plt.figure(1, clear=True, dpi=200)
plt.hist(high_risk, bins=bins, alpha=.3, color='r', label='High Risk')
plt.hist(low_risk, bins=bins, alpha=.3, color='b', label='Low Risk')
plt.legend()
plt.grid()
plt.xticks(np.arange(10,75,5))
plt.xlabel('Patient Ages')
plt.ylabel('Number of Patients')
plt.title('Age at time of delivery grouped\nby risk level')
plt.tight_layout()
plt.savefig('figure_all.png')
# %%
first_half_features = patient_ages[:508]
first_half_labels = patient_risk[:508]
second_half_features = patient_ages[508:]
second_half_labels = patient_risk[508:]
plt.figure(2, clear=True, dpi=200)
plt.clf()
plt.subplot(1,2,1)
plot_histogram(first_half_features, first_half_labels)
plt.title('Age at time of delivery grouped\nby risk level (First half of data)')
plt.subplot(1,2,2)
plot_histogram(second_half_features, second_half_labels)
plt.title('Age at time of delivery grouped\nby risk level (Second half of data)')
plt.savefig('figure_halved.png')

# %%
np.savetxt('data_features.txt', patient_ages, fmt='%s', delimiter=',')
np.savetxt('data_labels.txt', patient_risk, fmt='%s', delimiter=',')
reloaded_features = np.loadtxt('data_features.txt', delimiter=',', dtype='int')
reloaded_labels = np.loadtxt('data_labels.txt', delimiter=',', dtype='str')
is_features_equal = np.array_equal(reloaded_features, patient_ages)
is_labels_equal = np.array_equal(reloaded_labels, patient_risk)
if is_features_equal == True and is_labels_equal == True:
    print('Yay they are the same')
    
else:
    print('Boo they are not the same')



