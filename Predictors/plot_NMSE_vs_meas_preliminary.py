# -*- coding: utf-8 -*-
"""
Created on Thu May 26 10:26:24 2022

@author: EG
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py


fs = 250
prediction_horizon = 0

prediction_horizon_time = (prediction_horizon)/fs

snrs = [5,10,20] 
# snrs = [10] 


min_dist = 3
cell_radius = 2
velocity = 2
realizations = 1024
t = 30
scenario = 3
window = 1


legend_locs = [(0.7, 0.6), (0.68, 0.02)]
labels = ['Measurement horizon [s]', 'NMSE [dB]']

colors = ['tab:blue', 'tab:orange', 'tab:green', 'magenta', 'k']
linestyles = ['solid', 'dashed', 'dotted']
font = 25

test_scenarios = [(30,30,16,2),(30,30,16,20)]

fig, axs = plt.subplots(1, 2, sharex=True)
fig.set_figheight(6)
fig.set_figwidth(18)
plt.tight_layout=True
plines = []

for i, test_scenario in enumerate(test_scenarios):
    height, width, n_nodes, velocity = test_scenario
    
    savename = 'NMSE_vs_meas_hor_pred_hor_{}_size_{}x{}_nodes_{}_scenario_{}_preliminary_2.pdf'.format(prediction_horizon_time, height, width, n_nodes, scenario)

    for j in (range(len(snrs))):
        snr = snrs[j]
        hf = h5py.File('data/meas/NMSE_vs_meas_hor_and_pred_hor_pred_time_{}_vel_{}_size_{}x{}_nodes_{}_snr_{}_scenario_{}_preliminary_2.h5'.format(window, velocity, height, width, n_nodes, snr, scenario), 'r')
        results = np.array(hf.get('NMSE'))
        x = np.array(hf.get('x_axis'))
        hf.close()

        for k in range(len(results)):
            l1, = axs[i].plot(x, results[k,:,prediction_horizon], color = colors[k], linestyle = linestyles[j])
            plines.append(l1)
    
        axs[i].tick_params(labelsize = font)
        axs[i].set_title("Velocity {} [m/s]".format(velocity), fontsize=font)
        axs[i].grid(True)

snr_list = ["{} dB".format(snrs[0]),"{} dB".format(snrs[1]),"{} dB".format(snrs[2])]
snr_legend=plt.legend([plines[0],plines[5],plines[10]], snr_list, fontsize = font - 5, title = "SNR", title_fontsize = font-5, ncol=1, loc = legend_locs[0])
plt.legend(plines[:5],["AR(1)","AR(10)","AR(20)", "AR(50)", "LV"], fontsize = font - 5, title = "Predictors", title_fontsize = font-5, ncol=1, loc = legend_locs[1])
plt.gca().add_artist(snr_legend)
fig.text(0.51, 0.00, labels[0], fontsize = font, ha='center')
axs[0].set_ylabel(labels[1], fontsize = font) 
plt.savefig("figures\\perf_eval/"+savename, bbox_inches='tight')        
plt.show()