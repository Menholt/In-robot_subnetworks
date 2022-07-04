# -*- coding: utf-8 -*-
"""
Created on Thu May 26 18:50:35 2022

@author: EG
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py

fs = 250

measurement_horizon = 5
snrs = [5,10,20]
# snrs = [15] 

velocities = [2,20]


min_dist = 3
cell_radius = 2
realizations = 1024
t = 30
scenario = 3
window = 1
threshold = -5

legend_locs = [(0.66, 0.05), (0.67, 0.45)]
labels = ['Prediction horizon [ms]', 'NMSE [dB]']
colors = ['tab:blue', 'tab:orange', 'tab:green', 'k', 'magenta']
linestyles = ['solid', 'dashed', 'dotted']
font = 25

fig, axs = plt.subplots(1, 2, sharex=True)
fig.set_figheight(6)
fig.set_figwidth(18)
plt.tight_layout=True
plines = []

test_scenarios = [(30,30,16,2), (30,30,16,20)]
for j, test_scenario in enumerate(test_scenarios):
    height, width, n_nodes, velocity = test_scenario
    
    savename = 'NMSE_vs_pred_hor_meas_hor_{}_size_{}x{}_nodes_{}_scenario_{}_lv.pdf'.format(measurement_horizon, height, width, n_nodes, scenario)
    
    for i in (range(len(snrs))):
        snr = snrs[i]
        
        hf = h5py.File('data/pred/NMSE_vs_meas_hor_and_pred_hor_pred_time_{}_vel_{}_size_{}x{}_nodes_{}_snr_{}_scenario_{}_med_lv.h5'.format(window, velocity, height, width, n_nodes, snr, scenario), 'r')
        results = np.array(hf.get('NMSE'))[:,measurement_horizon-1,:]
        x = (np.arange(results.shape[1])+1)/fs*1000
        hf.close()
        for k in range(len(results)):
            l1, = axs[j].plot(x, results[k], color = colors[k], linestyle = linestyles[i])
            plines.append(l1)
        axs[j].tick_params(labelsize = font)
        axs[j].grid(True)
        axs[j].set_title("Velocity {} [m/s]".format(velocity), fontsize=font)
        axs[j].hlines(threshold, x[0], x[-1], color='k')

fig.text(0.51, 0.00, labels[0], fontsize = font, ha='center')
snr_list = ["{} [dB]".format(snrs[0]),"{} [dB]".format(snrs[1]),"{} [dB]".format(snrs[2])]
snr_legend=axs[1].legend([plines[0],plines[2],plines[4]], snr_list, fontsize = font - 5, title = "SNR", title_fontsize = font-5, ncol=1, loc = legend_locs[0])
axs[1].legend([plines[0],plines[1]],["AR(20)", "LV"], fontsize = font - 5, title = "Predictors", title_fontsize = font-5, ncol=1, loc = legend_locs[1])
axs[1].add_artist(snr_legend)
axs[0].set_ylabel(labels[1], fontsize = font)  
    # plt.xlabel(labels[0], fontsize = font)  
plt.savefig("figures\\perf_eval/"+savename, bbox_inches='tight')        
plt.show()

#%%

prediction_horizon = 1
prediction_horizon_time = (prediction_horizon)/fs

savename = 'NMSE_CDF_pred_hor_{}_meas_hor_{}_size_{}x{}_nodes_{}_scenario_{}_lv.pdf'.format(prediction_horizon_time, measurement_horizon, height, width, n_nodes, scenario)
legend_locs = [(0.01, 0.31), (0.01, 0.7)]
labels = ['NMSE [dB]', 'Prob(NMSE<abscissa)']

fig, axs = plt.subplots(1, 2, sharex=True)
fig.set_figheight(6)
fig.set_figwidth(18)
plt.tight_layout=True
plines = []

test_scenarios = [(30,30,16,2), (30,30,16,20)]

for j, test_scenario in enumerate(test_scenarios):
    height, width, n_nodes, velocity = test_scenario

    for i in (range(len(snrs))):
        snr = snrs[i]
        
        hf = h5py.File('data/cdf/NMSE_CDF_all_pred_hor_meas_hor_{}_pred_time_{}_vel_{}_size_{}x{}_nodes_{}_snr_{}_scenario_{}_lv.h5'.format(measurement_horizon, window, velocity, height, width, n_nodes, snr, scenario), 'r')
        results = np.array(hf.get('CDFs'))[:,prediction_horizon-1]
        x = np.array(hf.get('x_axis'))
        hf.close()

        for k in range(len(results)):
            l1, = axs[j].plot(x, results[k], color = colors[k], linestyle = linestyles[i])
            plines.append(l1)
        axs[j].grid(True)
        axs[j].tick_params(labelsize = font)

        axs[j].vlines(threshold, 0, 1, color='k')

        axs[j].set_title("Velocity {} [m/s]".format(velocity), fontsize=font)

fig.text(0.51, 0.00, labels[0], fontsize = font, ha='center')
snr_list = ["{} [dB]".format(snrs[0]),"{} [dB]".format(snrs[1]),"{} [dB]".format(snrs[2])]
snr_legend=plt.legend([plines[0],plines[2],plines[4]], snr_list, fontsize = font - 5, title = "SNR", title_fontsize = font-5, ncol=1, loc = legend_locs[0])
plt.legend([plines[0],plines[1]],["AR(20)", "LV"], fontsize = font - 5, title = "Predictors", title_fontsize = font-5, ncol=1, loc = legend_locs[1])
plt.gca().add_artist(snr_legend)
# plt.xlabel(labels[0] ,fontsize = font)    
axs[0].set_ylabel(labels[1], fontsize = font)  
plt.savefig("figures\\perf_eval/"+savename, bbox_inches='tight')        
plt.show()

#%%

savename = 'NMSE_vs_snr_pred_hor_{}_meas_hor_{}_size_{}x{}_nodes_{}_scenario_{}_lv.pdf'.format(prediction_horizon_time, measurement_horizon, height, width, n_nodes, scenario)

labels = ['SNR', 'NMSE [dB]']
fig, axs = plt.subplots(1, 2, sharex=True)
fig.set_figheight(6)
fig.set_figwidth(18)
plt.tight_layout=True
plines = []

test_scenarios = [(30,30,16,2), (30,30,16,20)]
for i, test_scenario in enumerate(test_scenarios):
    height, width, n_nodes, velocity = test_scenario
        

    fig.text(0.51, 0.00, labels[0], fontsize = font, ha='center')
    
    hf = h5py.File('data/snr/NMSE_vs_snr_all_pred_hor_meas_hor_{}_pred_time_{}_vel_{}_size_{}x{}_nodes_{}_scenario_{}.h5'.format(measurement_horizon, window, velocity, height, width, n_nodes, scenario), 'r')
    results = np.array(hf.get('NMSE'))
    x = np.array(hf.get('x_axis'))
    hf.close()
    # results[results>5] = np.nan

        
    for k in range(len(results)):
        l1 = axs[i].plot(x, results[k, :, prediction_horizon-1], color = colors[k])
        plines.append(l1)
    axs[i].tick_params(labelsize = font)
    # axs[i].set_xlabel(labels[0], fontsize = font)    
    axs[i].grid(True)
    axs[i].hlines(threshold, x[0], x[-1], color='k')
    axs[i].set_title("Velocity {} [m/s]".format(velocity), fontsize=font)

# plt.legend([plines[0],plines[-1]],["2 m/s", "20 m/s"], fontsize = font - 5, title = "Velocity", title_fontsize = font-5, ncol=1, loc = 1)
plt.legend(["AR(20)", "LV"], fontsize = font - 5, title = "Predictors", title_fontsize = font-5, ncol=1, loc = 1)
axs[0].set_ylabel(labels[1], fontsize = font)  
# plt.gca().add_artist(legend)
# plt.title("Room size {}x{}, n_nodes {}, velocity {}, predicton horizon {}".format(height, width, n_nodes, velocity, (i+1)/fs), fontsize=font)

plt.savefig("figures\\perf_eval\\"+savename, bbox_inches='tight')        
plt.show()



