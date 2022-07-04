# -*- coding: utf-8 -*-
"""
Created on Fri May 27 09:29:57 2022

@author: EG
"""
import numpy as np
from statsmodels.regression.linear_model import yule_walker

def ar_predictor(interference_meas, interference_true, p, max_prediction_horizon, measurement_horizon, fs, window, time_trace=False):
    normalize = np.mean(interference_true[int(measurement_horizon*fs):int(measurement_horizon*fs)+int(window*fs)]**2)
    # normalize = 1

    I_hat = np.zeros((int(window*fs), max_prediction_horizon))
    error = np.zeros((int(window*fs), max_prediction_horizon))
    NMSE = np.zeros((int(window*fs), max_prediction_horizon))
    
    beta, sigma = yule_walker(interference_meas[:int(measurement_horizon*fs)], order = p)
    for i in range(int(window*fs)):
        interference_temp = np.copy(interference_meas)
        for j in range(max_prediction_horizon):
            I_hat[i,j] = np.inner(np.flip(beta), interference_temp[int(measurement_horizon*fs)+i+j-p:int(measurement_horizon*fs)+i+j])
            
            interference_temp[int(measurement_horizon*fs)+i+j] = I_hat[i,j]

        error[i] = I_hat[i] - interference_true[int(measurement_horizon*fs)+i:int(measurement_horizon*fs)+max_prediction_horizon+i]

    NMSE = 10*np.log10(np.mean(error**2, axis=0)/normalize)
    if time_trace==True:
        return NMSE, I_hat
    else:
        return NMSE

def lv_predictor(interference_meas, interference_true, max_prediction_horizon, measurement_horizon, fs, window):
    
    I_hat = np.zeros((int(window*fs), max_prediction_horizon))
    error = np.zeros((int(window*fs), max_prediction_horizon))
    NMSE = np.zeros((int(window*fs), max_prediction_horizon))
    
    # normalize = np.mean(interference[int(measurement_horizon*fs):int(measurement_horizon*fs)+int(window*fs)]**2)
    normalize = np.mean(interference_true[int(measurement_horizon*fs):int(measurement_horizon*fs)+int(window*fs)]**2)

    # normalize = 1
    for i in range(int(window*fs)):
        for j in range(max_prediction_horizon):
            I_hat[i,:] = interference_meas[int(measurement_horizon*fs)+i-1]

        # error[i] = I_hat[i] - interference[int(measurement_horizon*fs)+i:int(measurement_horizon*fs)+max_prediction_horizon+i]
        error[i] = I_hat[i] - interference_true[int(measurement_horizon*fs)+i:int(measurement_horizon*fs)+max_prediction_horizon+i]

    NMSE = 10*np.log10(np.mean(error**2, axis=0)/normalize)
    return NMSE#, I_hat 

def mv_predictor(interference_meas, interference_true, max_prediction_horizon, measurement_horizon, fs, window):
    
    I_hat = np.ones((int(window*fs), max_prediction_horizon))*np.mean(interference_meas[:int(measurement_horizon*fs)])
    # I_hat = np.ones((int(window*fs), max_prediction_horizon))*np.mean(interference_meas[:])
    error = np.zeros((int(window*fs), max_prediction_horizon))
    NMSE = np.zeros((int(window*fs), max_prediction_horizon))
    
    normalize = np.mean(interference_true[int(measurement_horizon*fs):int(measurement_horizon*fs)+int(window*fs)]**2)
    
    for i in range(int(window*fs)):
        # error[i] = I_hat[i] - interference[int(measurement_horizon*fs)+i:int(measurement_horizon*fs)+max_prediction_horizon+i]
        error[i] = I_hat[i] - interference_true[int(measurement_horizon*fs)+i:int(measurement_horizon*fs)+max_prediction_horizon+i]

    NMSE = 10*np.log10(np.mean(error**2, axis=0)/normalize)
    return NMSE#, I_hat 
