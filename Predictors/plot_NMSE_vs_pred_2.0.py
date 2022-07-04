# -*- coding: utf-8 -*-
"""
Created on Sun May 22 13:57:09 2022

@author: EG
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py


fs = 250
prediction_horizon = 5

prediction_horizon_time = (prediction_horizon)/fs
x=np.arange(1,prediction_horizon+1)/fs

snrs = [5,10] 
# snrs = [10] 


min_dist = 3
cell_radius = 2
velocity = 2
realizations = 1024
t = 30
scenario = 3
window = 1


legend_locs = [(0.85, 0.7), (0.85, 0.12)]
labels = ['Prediction horizon [s]', 'NMSE [dB]']

colors = ['tab:blue', 'tab:orange', 'tab:green', 'k', 'magenta']
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'k', 'magenta', 'cyan', 'gray', 'purple']

linestyles = ['solid', 'dashed']
font = 25


test_scenarios = [(30,30,16,2), (60,60,16,2), (60,60,32,2), (30,30,16,20), (60,60,32,20), (60,60,16,20)]
test_scenarios = [(30,30,16,2), (30,30,16,20)]
for test_scenario in test_scenarios:
    height, width, n_nodes, velocity = test_scenario
    
    savename = 'NMSE_vs_pred_hor_vel_{}_size_{}x{}_nodes_{}_scenario_{}.pdf'.format(velocity, height, width, n_nodes, scenario)

    fig, axs = plt.subplots(10, 1, sharex=False)
    fig.set_figheight(10*6) # 20 is number of measurement horizons
    fig.set_figwidth(18)
    fig.set_tight_layout('tight')
    plines = [[],[]]
    for j in (range(len(snrs))):
        snr = snrs[j]
        hf = h5py.File('data/pred/NMSE_vs_meas_hor_and_pred_hor_pred_time_{}_vel_{}_size_{}x{}_nodes_{}_snr_{}_scenario_{}.h5'.format(window, velocity, height, width, n_nodes, snr, scenario), 'r')
        results = np.array(hf.get('NMSE'))
        measurement_horizons = np.array(hf.get('x_axis'))
        hf.close()

        for i in range(len(measurement_horizons)):

            for k in range(len(results)):
                l1, = axs[i].plot(x, results[k,i], color = colors[k], linestyle = linestyles[j])
                plines[j].append(l1)
        
            if j==1:
                snr_legend=axs[i].legend([plines[0][0],plines[1][0]], ["{} dB".format(snrs[0]),"{} dB".format(snrs[1])], fontsize = font - 5, title = "SNR", title_fontsize = font-5, ncol=1, loc = legend_locs[0])
                # axs[i].legend(plines[0][:8],["AR(1)", "AR(2)", "AR(5)","AR(10)","AR(20)", "AR(50)", "AR(100)", "LV"], fontsize = font - 5, title = "Predictors", title_fontsize = font-5, ncol=1, loc = legend_locs[1])
                axs[i].legend(plines[0][:5],["AR(1)", "AR(10)", "AR(20)", "AR(50)", "LV"], fontsize = font - 5, title = "Predictors", title_fontsize = font-5, ncol=1, loc = legend_locs[1])
                # plt.gca().add_artist(snr_legend)
                axs[i].add_artist(snr_legend)
                axs[i].tick_params(labelsize = font)
                axs[i].set_xlabel(labels[0], fontsize = font)    
                axs[i].set_ylabel(labels[1], fontsize = font) 
                
                axs[i].set_title("Room size {}x{}, #subnetworks {}, velocity {}[m/s], measurement horizon {}[s]".format(height, width, n_nodes, velocity, i+1), fontsize=font)
    plt.savefig("figures\\pred/"+savename)#, bbox_inches='tight')        
    plt.show()