# -*- coding: utf-8 -*-
"""
Created on Sat May 21 10:47:41 2022

@author: EG
"""


import numpy as np
from tqdm import tqdm
import h5py
from predictors import ar_predictor, lv_predictor, mv_predictor



fs = 250
# measurement_horizon = 29
prediction_horizon_time = 0.02
prediction_horizon = int(prediction_horizon_time*fs)

# measurement_horizons = np.linspace(0.5, 20, 40)
measurement_horizon = 5

orders = [20]

snrs = np.arange(26)
# snrs = [15] 


min_dist = 3
cell_radius = 2
velocity = 2
realizations = 1024
t = 30
scenario = 3
window = 1

#%%

# test_scenarios = [(30,30,16,2),(60,60,16,2),(60,60,32,2)]
test_scenarios = [(30,30,16,2),(60,60,16,2),(60,60,32,2),(60,60,32,20),(60,60,16,20),(30,30,16,20)]
for test_scenario in test_scenarios:
    height, width, n_nodes, velocity = test_scenario
    
    hf = h5py.File('../Network_model/data\\performance_evaluation_scenarios\\interference_realization_{}_minDist_{}_cell_radius_{}_vel_{}_time_{}_fs_{}_size_{}x{}_nodes_{}_scenario_{}.h5'.format(realizations, min_dist, cell_radius, velocity, t, fs, height, width, n_nodes, scenario), 'r')
    interferences_true = np.array(hf.get('interference'))[:, :int(measurement_horizon*fs)+prediction_horizon+1 + int(fs*window)]
    hf.close()
    
    temp = interferences_true
    for j in range(len(interferences_true)):
        if np.mean(10*np.log10(interferences_true[len(interferences_true)-j-1,:])) <= -150:
            temp = np.delete(temp, len(interferences_true)-j-1, 0)
    interferences_true = temp
    
    NMSE = np.zeros((len(orders)+1, len(snrs), prediction_horizon))

    for ii in tqdm(range(len(snrs))):
        snr = snrs[ii]
        hf = h5py.File('../Network_model/data\\performance_evaluation_scenarios\\interference_realization_{}_minDist_{}_cell_radius_{}_vel_{}_time_{}_fs_{}_size_{}x{}_nodes_{}_snr_{}_scenario_{}.h5'.format(realizations, min_dist, cell_radius, velocity, t, fs, height, width, n_nodes, snr, scenario), 'r')
        interferences_meas = np.array(hf.get('interference'))[:, :int(measurement_horizon*fs)+prediction_horizon+1 + int(fs*window)]
        hf.close()

        temp = interferences_meas
        for j in range(len(interferences_meas)):
            if np.mean(10*np.log10(interferences_meas[len(interferences_meas)-j-1,:])) <= -150:
                temp = np.delete(temp, len(interferences_meas)-j-1, 0)
        interferences_meas = temp #10*np.log10(temp)
        
        for i, p in enumerate(orders):
            for j in (range(len(interferences_meas))):
                NMSE[i,ii] += ar_predictor(interferences_meas[j], interferences_true[j], p, prediction_horizon, measurement_horizon, fs, window)/len(interferences_meas)
        for j in (range(len(interferences_meas))):
            NMSE[-1, ii] += lv_predictor(interferences_meas[j], interferences_true[j], prediction_horizon, measurement_horizon, fs, window)/len(interferences_meas)
    
    hf = h5py.File('data/snr/NMSE_vs_snr_all_pred_hor_meas_hor_{}_pred_time_{}_vel_{}_size_{}x{}_nodes_{}_scenario_{}.h5'.format(measurement_horizon, window, velocity, height, width, n_nodes, scenario), 'w')
    hf.create_dataset('NMSE', data = NMSE)
    hf.create_dataset('x_axis', data = snrs)
    hf.close()
