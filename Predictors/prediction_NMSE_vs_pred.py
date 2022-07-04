# -*- coding: utf-8 -*-
"""
Created on Mon May  9 14:45:08 2022

@author: EG
"""

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import h5py
from statsmodels.regression.linear_model import yule_walker


def ar_predictor(interference_meas, interference_true, p, max_prediction_horizon, measurement_horizon, fs, window):
    normalize = np.mean(interference_true[int(measurement_horizon*fs):int(measurement_horizon*fs)+int(window*fs)]**2)

    I_hat = np.zeros((int(window*fs), max_prediction_horizon))
    error = np.zeros((int(window*fs), max_prediction_horizon))
    NMSE = np.zeros((int(window*fs), max_prediction_horizon))
    
    beta, sigma = yule_walker(interference_meas[:int(measurement_horizon*fs)], order = p)
    for i in range(int(window*fs)):
        interference_temp = np.copy(interference_meas)
        for j in range(max_prediction_horizon):
            for n in range(p):
                I_hat[i,j] += beta[n]*interference_temp[int(measurement_horizon*fs)+i+j-(n+1)]
            interference_temp[int(measurement_horizon*fs)+i+j] = I_hat[i,j]

        error[i] = I_hat[i] - interference_true[int(measurement_horizon*fs)+i:int(measurement_horizon*fs)+max_prediction_horizon+i]

    NMSE = 10*np.log10(np.mean(error**2, axis=0)/normalize)
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


fs = 250
# measurement_horizon = 29
max_prediction_horizon_time = 0.1
max_prediction_horizon = int(max_prediction_horizon_time*fs)    

# measurement_horizons = np.linspace(0.5, 20, 40)
measurement_horizon = 20

orders = [1, 10, 20]

snrs = [5, 10] 
# snrs = [15] 


min_dist = 3
cell_radius = 2

realizations = 1024
t = 30
scenario = 3
window = 1
height = 30
width = 30
n_nodes = 16
velocity = 20
#%%

test_scenarios = [(30,30,16,2), (60,60,16,2), (60,60,32,2), (30,30,16,20), (60,60,32,20), (60,60,16,20)]
test_scenarios = [(60,60,32,20)]
test_scenarios = [(60,60,16,2)]
for test_scenario in test_scenarios:
    height, width, n_nodes, velocity = test_scenario
    
    font = 25
    plt.figure(figsize=(18,6))
    plt.tight_layout=True
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'magenta']
    linestyles = ['solid', 'dashed', 'dotted']
    plines = []
    
    for ii in (range(len(snrs))):
        snr = snrs[ii]
        hf = h5py.File('../Network_model/data\\performance_evaluation_scenarios\\interference_realization_{}_minDist_{}_cell_radius_{}_vel_{}_time_{}_fs_{}_size_{}x{}_nodes_{}_snr_{}_scenario_{}.h5'.format(realizations, min_dist, cell_radius, velocity, t, fs, height, width, n_nodes, snr, scenario), 'r')
        interferences = np.array(hf.get('interference'))[:, :measurement_horizon*fs+max_prediction_horizon+1 + int(fs*window)]
        hf.close()

        temp = interferences
        for j in range(len(interferences)):
            if np.mean(10*np.log10(interferences[len(interferences)-j-1,:])) <= -150:
                temp = np.delete(temp, len(interferences)-j-1, 0)
        interferences = temp #10*np.log10(temp)
        NMSE = np.zeros((len(orders)+1, max_prediction_horizon))
        for i, p in enumerate(orders):
            for interference in tqdm(interferences):
                NMSE[i,:] += ar_predictor(interference, p, max_prediction_horizon, measurement_horizon, fs, window)/len(interferences)    
            l1, = plt.plot((np.arange(max_prediction_horizon)+1)/fs, NMSE[i], color = colors[i], linestyle = linestyles[ii])
            plines.append(l1)
        for interference in tqdm(interferences):
            NMSE[-1] += lv_predictor(interference, max_prediction_horizon, measurement_horizon, fs, window)/len(interferences)
        l1, = plt.plot((np.arange(max_prediction_horizon)+1)/fs, NMSE[-1], color = colors[-1], linestyle = linestyles[ii])
        plines.append(l1)
        plt.legend(orders+["LV"], fontsize=font-5, title = "Ar order", title_fontsize = font-5)
        plt.tick_params(labelsize = font - 5)
      
        plt.ylabel('NMSE [dB]', fontsize = font)
        
        # plt.show()
        x = (np.arange(max_prediction_horizon)+1)/fs
        hf = h5py.File('data/pred/NMSE_vs_pred_hor_meas_hor_{}_pred_time_{}_vel_{}_size_{}x{}_nodes_{}_snr_{}_scenario_{}.h5'.format(measurement_horizon, window, velocity, height, width, n_nodes, snr, scenario), 'w')
        hf.create_dataset('NMSE', data = NMSE)
        hf.create_dataset('x_axis', data = x)
        hf.close()
    snr_legend=plt.legend([plines[0],plines[4]], ["{} dB".format(snrs[0]),"{} dB".format(snrs[1])], fontsize = font - 5, title = "SNR", title_fontsize = font-5, ncol=2, loc=(0.42,0.01))
    plt.legend(plines[:4],["AR(1)","AR(10)","AR(20)", "LV"], fontsize = font - 5, title = "Predictors", title_fontsize = font-5, ncol=2, loc = (0.7,0.01))
    plt.gca().add_artist(snr_legend)
    plt.title('Size {}x{}, nodes {}, velocity {}m/s, measurement horizon {}s'.format(height, width, n_nodes, velocity, measurement_horizon), fontsize=font)
    plt.xlabel('Prediction horizon [s]', fontsize = font)
    plt.ylabel('NMSE [dB]', fontsize = font)    
    plt.xticks(fontsize = font)
    plt.yticks(fontsize = font)
    
    # if snr_on:
    # plt.savefig("figures\\NMSE_vs_pred_hor_meas_hor_{}_vel_{}_size_{}x{}_nodes_{}_scenario_{}.pdf".format(measurement_horizon, velocity, height, width, n_nodes, scenario))        
    # else:
    #     plt.savefig("figures\\NMSE_vs_pred_hor_meas_hor_{}_vel_{}_noSNR_scenario_{}.pdf".format(measurement_horizon, velocity, scenario))
    plt.show()