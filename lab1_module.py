# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 17:03:32 2023

@author: bentn
"""
import numpy as np
import matplotlib.pyplot as plt
def plot_histogram(features, labels):
    '''
    This function creates a hist

    Parameters
    ----------
    features : TYPE
        DESCRIPTION.
    labels : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    high_risk_patient_mask = labels == 'high risk'
    mid_risk_patient_mask = labels == 'mid risk'
    low_risk_patient_mask = labels == 'low risk'
    high_risk = features[high_risk_patient_mask]
    mid_risk = features[mid_risk_patient_mask]
    low_risk = features[low_risk_patient_mask]
    bins = np.arange(9.5,68.5, 1)
    plt.hist(high_risk, bins=bins, alpha=.3, color='r', label='High Risk')
    plt.hist(low_risk, bins=bins, alpha=.3, color='b', label='Low Risk')
    plt.legend()
    plt.grid()
    plt.xticks(np.arange(10,75,5))
    plt.xlabel('Patient Ages')
    plt.ylabel('Number of Patients')
    plt.title('Age at time of delivery grouped\nby risk level')
    plt.tight_layout()