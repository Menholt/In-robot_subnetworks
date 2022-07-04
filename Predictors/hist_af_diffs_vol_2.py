# -*- coding: utf-8 -*-
"""
Created on Wed May 25 13:17:22 2022

@author: EG
"""


import numpy as np
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt

fs = 250


min_dist = 3
cell_radius = 2
velocity = 2
scenario = 3
realizations = 1024
t = 30

bins = np.linspace(0, 15, 91)



font = 25

test_scenarios = [(30,30,16,2), (60,60,16,2), (60,60,32,2)] #, (30,30,16,20), (60,60,32,20), (60,60,16,20)]
# test_scenarios = [(30,30,16,20), (60,60,32,20), (60,60,16,20)]
test_scenarios = [(30,30,16,20)]
for test_scenario in test_scenarios:
    height, width, n_nodes, velocity = test_scenario
    
    plt.figure(figsize=(18,6))
    plt.tight_layout=True
    
    hf = h5py.File('../Network_model/data\\performance_evaluation_scenarios\\interference_realization_{}_minDist_{}_cell_radius_{}_vel_{}_time_{}_fs_{}_size_{}x{}_nodes_{}_scenario_{}.h5'.format(realizations, min_dist, cell_radius, velocity, t, fs, height, width, n_nodes, scenario), 'r')
    interference = np.array(hf.get('interference'))
    hf.close()
    
    temp = interference
    for j in range(len(interference)):
        if np.mean(10*np.log10(interference[len(interference)-j-1,:])) <= -150:
            temp = np.delete(temp, len(interference)-j-1, 0)
    interference = temp
    
    diffs = np.abs(np.diff(10*np.log10(interference), axis = 1))
    
    mid = np.median(diffs)
    
    a,_, __ = plt.hist(diffs.reshape(-1), bins = bins, density = True)
    plt.vlines(mid, 0, np.max(a), "k", linewidth = 5)
    # plt.title("Room size {}x{}, n_nodes {}, velocity {}".format(height, width, n_nodes, velocity), fontsize=font)
    plt.ylabel("Normalized frequency", fontsize = font)
    plt.xlabel("Difference [dB]", fontsize = font)
    plt.tick_params(labelsize = font)
    # axs[int(i//2),int(i%2)].legend(("Median = {:.3F}".format(mid), "Differences"), fontsize = font-5)
    plt.legend((["Median = {:.3F}".format(mid)]), fontsize = font-5)
plt.savefig("figures/histogram_diffs_vel_20.pdf", bbox_inches = "tight")        
plt.show()