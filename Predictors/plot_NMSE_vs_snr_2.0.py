# -*- coding: utf-8 -*-
"""
Created on Mon May 23 14:43:55 2022

@author: EG
"""


import numpy as np
import matplotlib.pyplot as plt
import h5py


fs = 250
prediction_horizon = 25
prediction_horizon_time = (prediction_horizon)/fs
measurement_horizon = 5


snrs = [5,10] 


min_dist = 3
cell_radius = 2
velocity = 2
realizations = 1024
t = 30
scenario = 3
window = 1

legend_locs = [1,1]
labels = ["SNR [dB]", 'NMSE [dB]']

colors = ['tab:blue', 'tab:orange', 'tab:green', 'k', 'magenta']
linestyles = ['solid', 'dashed']
font = 25


test_scenarios = [(30,30,16,2), (60,60,16,2), (60,60,32,2), (30,30,16,20), (60,60,32,20), (60,60,16,20)]
test_scenarios = [(30,30,16,20), (60,60,32,20), (60,60,16,20)]
for test_scenario in test_scenarios:
    height, width, n_nodes, velocity = test_scenario
        
    fig, axs = plt.subplots(prediction_horizon, 1, sharex=True)
    fig.set_figheight(prediction_horizon*6)
    fig.set_figwidth(18)
    plines = []
    
    hf = h5py.File('data/snr/NMSE_vs_snr_all_pred_hor_meas_hor_{}_pred_time_{}_vel_{}_size_{}x{}_nodes_{}_scenario_{}.h5'.format(measurement_horizon, window, velocity, height, width, n_nodes, scenario), 'r')
    results = np.array(hf.get('NMSE'))
    x = np.array(hf.get('x_axis'))
    hf.close()
    savename = 'NMSE_vs_snr_meas_hor_{}_vel_{}_size_{}x{}_nodes_{}_scenario_{}.pdf'.format(measurement_horizon, velocity, height, width, n_nodes, scenario)
    results[results>5] = np.nan
    
    for i in range(prediction_horizon):
        for k in range(len(results)):
            axs[i].plot(x, results[k,:,i], color = colors[k])
        
        # snr_legend=axs[i].legend([plines[0],plines[4]], ["{} dB".format(snrs[0]),"{} dB".format(snrs[1])], fontsize = font - 5, title = "SNR", title_fontsize = font-5, ncol=1, loc = legend_locs[0])
        axs[i].legend(["AR(1)","AR(10)","AR(20)", "LV"], fontsize = font - 5, title = "Predictors", title_fontsize = font-5, ncol=1, loc = legend_locs[1])
        # plt.gca().add_artist(snr_legend)
        axs[i].tick_params(labelsize = font)
        axs[i].set_xlabel(labels[0], fontsize = font)    
        axs[i].set_ylabel(labels[1], fontsize = font)  
        axs[i].set_title("Room size {}x{}, n_nodes {}, velocity {}, predicton horizon {}".format(height, width, n_nodes, velocity, (i+1)/fs), fontsize=font)
    plt.savefig("figures\\snr\\"+savename, bbox_inches='tight')        
    plt.show()