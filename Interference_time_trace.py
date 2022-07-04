# -*- coding: utf-8 -*-
"""
Created on Thu May 26 09:38:18 2022

@author: lukas
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py


min_dist = 3
cell_radius = 2
velocity = 2
realizations = 1024
t = 30
scenario = 3
fs = 250


snrs = [5,10,20] 
x = np.linspace(0, 1, fs)
x_ticks = [0,  0.5,  1]

font = 25
labels = ['time [s]', 'Interference power [dB]']

test_scenarios = [(30,30,16,2), (30,30,16,20)]

for test_scenario in test_scenarios:
    height, width, n_nodes, velocity = test_scenario
    
    fig, axs = plt.subplots(1, 3, sharey=True)
    fig.set_figheight(6) 
    fig.set_figwidth(18)
    fig.set_tight_layout('tight')
    
    for i in (range(len(snrs))):
        snr = snrs[i]
        hf = h5py.File('data\\performance_evaluation_scenarios\\interference_realization_{}_minDist_{}_cell_radius_{}_vel_{}_time_{}_fs_{}_size_{}x{}_nodes_{}_snr_{}_scenario_{}.h5'.format(realizations, min_dist, cell_radius, velocity, t, fs, height, width, n_nodes, snr, scenario), 'r')
        interference = np.array(hf.get('interference'))[0,:fs]
        hf.close()
        axs[i].plot(x, 10*np.log10(interference))
        axs[i].tick_params(labelsize = font)
        axs[i].set_xticks(x_ticks)
        axs[i].set_title("SNR: {}".format(snr), fontsize = font)
    axs[0].set_ylabel(labels[1], fontsize = font)
    axs[1].set_xlabel(labels[0], fontsize = font)
    # plt.savefig("figures/interference_time_traces_vel_{}.pdf".format(test_scenario[-1]))#, bbox_inches='tight')        
    plt.show()

