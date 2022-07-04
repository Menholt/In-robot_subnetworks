# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 09:38:21 2022

@author: EG
"""

import numpy as np
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt

fs = 250
min_dist = 3
cell_radius = 2
scenario = 3 
realizations = 1024
t = 30

test_scenarios = [(30,30,16,2),(60,60,16,2),(60,60,32,2), (30,30,16,20)]
test_scenarios = [(60,60,32,20)]
test_scenarios = [(60,60,16,20)]
snrs = np.linspace(0, 25, 26)

for test_scenario in tqdm(test_scenarios):
    height, width, n_nodes, vel = test_scenario
    
    # hf = h5py.File('data/extra_scenarios/interference_realization_{}_minDist_{}_cell_radius_{}_vel_{}_time_{}_fs_{}_scenario_{}.h5'.format(realizations, min_dist, cell_radius, velocity, t, fs, scenario), 'r')
    hf = h5py.File('data\\performance_evaluation_scenarios\\interference_realization_{}_minDist_{}_cell_radius_{}_vel_{}_time_{}_fs_{}_size_{}x{}_nodes_{}_scenario_{}.h5'.format(realizations, min_dist, cell_radius, vel, t, fs, height, width, n_nodes, scenario), 'r')
    interference = np.array(hf.get('interference'))
    hf.close()
    for snr in snrs:
        
        new_interference = np.zeros((realizations, fs*t))
        for i in range(realizations):
            noise_variance = np.mean(interference[i])*10**(-snr/10)
            noise = np.sqrt(noise_variance)*np.random.normal(0,1,fs*t)
            new_interference[i] = interference[i]+noise**2
    
        plt.plot((10*np.log10(new_interference[0,:100])))
        plt.plot((10*np.log10(interference[0,:100])))
        plt.show()
        print(10*np.log10(np.mean(interference[-1])/np.mean(np.abs(noise)**2)))
        # print(10*np.log10(np.mean(noise**2)))

        # hf = h5py.File('data/extra_scenarios/interference_realization_{}_minDist_{}_cell_radius_{}_vel_{}_time_{}_fs_{}_snr_{}_scenario_{}.h5'.format(realizations, min_dist, cell_radius, velocity, t, fs, snr, scenario), 'w')
        hf = h5py.File('data\\performance_evaluation_scenarios\\interference_realization_{}_minDist_{}_cell_radius_{}_vel_{}_time_{}_fs_{}_size_{}x{}_nodes_{}_snr_{}_scenario_{}.h5'.format(realizations, min_dist, cell_radius, vel, t, fs, height, width, n_nodes, int(snr), scenario), 'w')
        hf.create_dataset('interference', data = new_interference)
        hf.close()
            
#%%

# scenario = 1
# snr = 5

# hf = h5py.File('data/scenario_{}/interference_realization_{}_minDist_{}_cell_radius_{}_vel_{}_time_{}_fs_{}_scenario_{}.h5'.format(scenario, realizations, min_dist, cell_radius, velocity, t, fs, scenario), 'r')
# interference = np.array(hf.get('interference'))
# hf.close()


# hf = h5py.File('data/scenario_{}/interference_realization_{}_minDist_{}_cell_radius_{}_vel_{}_time_{}_fs_{}_snr_{}_scenario_{}.h5'.format(scenario, realizations, min_dist, cell_radius, velocity, t, fs, snr, scenario), 'r')
# interference_snr = np.array(hf.get('interference'))
# hf.close()

# factor = int(10000/fs)
# plt.plot(10*np.log10(interference_snr[0, :int(5000/factor)]))
# plt.plot(10*np.log10(interference[0, :int(5000/factor)]))
# plt.show()

