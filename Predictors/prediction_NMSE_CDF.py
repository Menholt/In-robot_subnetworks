# -*- coding: utf-8 -*-
"""
Created on Tue May 17 13:08:54 2022

@author: EG
"""

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import h5py
from predictors import ar_predictor, lv_predictor, mv_predictor



fs = 250
# measurement_horizon = 29
prediction_horizon_time = 0.02
prediction_horizon = int(prediction_horizon_time*fs) 

# measurement_horizons = np.linspace(0.5, 20, 40)
measurement_horizon = 5

# orders = [1, 10, 20]
orders = [20]

snrs = [5,10,20] 
# snrs = [15] 


min_dist = 3
cell_radius = 2
velocity = 2
realizations = 1024
t = 30
scenario = 3
window = 1

#%%

test_scenarios = [(30,30,16,2),(60,60,16,2),(60,60,32,2), (30,30,16,20), (60,60,16,20),(60,60,32,20)]
# test_scenarios = [(30,30,16,20), (30,30,16,2)]
# test_scenarios = [(30,30,16,2),(60,60,16,2),(60,60,32,2)]
# test_scenarios = [(60,60,16,20)]
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
        
        NMSE = np.zeros((len(orders), len(interferences_meas), prediction_horizon))
        for i, p in enumerate(orders):
            for j in (range(len(interferences_meas))):
                NMSE[i,j] = ar_predictor(interferences_meas[j], interferences_true[j], p, prediction_horizon, measurement_horizon, fs, window)
        # for j in (range(len(interferences_meas))):
            # NMSE[-1, j] = lv_predictor(interferences_meas[j], interferences_true[j], prediction_horizon, measurement_horizon, fs, window)
            # NMSE[-1, j] = mv_predictor(interferences_true[j], interferences_true[j], prediction_horizon, measurement_horizon, fs, window)
    
    
        hists = np.zeros((len(NMSE), prediction_horizon, 64*2))
        bins = np.linspace(-30, 30, 64*2+1)
        
        for i in range(len(NMSE)):
            for j in range(prediction_horizon):
                hists[i, j], _ = np.histogram(NMSE[i, :, j], bins = bins, density = True)
            
        x = (bins[:-1]+bins[1:])/2
        cdfs = np.cumsum(hists, axis = 2)*(bins[1]-bins[0])
        
        hf = h5py.File('data/cdf/NMSE_CDF_all_pred_hor_meas_hor_{}_pred_time_{}_vel_{}_size_{}x{}_nodes_{}_snr_{}_scenario_{}_lukas.h5'.format(measurement_horizon, window, velocity, height, width, n_nodes, snr, scenario), 'w')
        hf.create_dataset('CDFs', data = cdfs)
        hf.create_dataset('x_axis', data = x)
        hf.close()
        
    #     for i in range(len(NMSE)):
    #         l1, = plt.plot(x, cdfs[i], color = colors[i], linestyle = linestyles[ii])
    #         plines.append(l1)
        
    

    # snr_legend=plt.legend([plines[0],plines[4]], ["{} dB".format(snrs[0]),"{} dB".format(snrs[1])], fontsize = font - 5, title = "SNR", title_fontsize = font-5, ncol=2, loc=(0.01,0.8))
    # plt.legend(plines[:4],["AR(1)","AR(5)","AR(10)", "LV"], fontsize = font - 5, title = "Predictors", title_fontsize = font-5, ncol=2, loc = (0.01,0.51))
    # plt.gca().add_artist(snr_legend)
    # plt.tick_params(labelsize = font - 5)
    # plt.ylabel('Prob(NMSE<abscissa)', fontsize = font)    
    # plt.xlabel('NMSE [dB]', fontsize = font)
    # plt.savefig("figures\\NMSE_CDF_pred_hor_{}_meas_hor_{}_vel_{}_size_{}x{}_nodes_{}_snr_{}_scenario_{}.pdf".format(prediction_horizon_time, measurement_horizon, velocity, height, width, n_nodes, snr, scenario))        
    # plt.show() 