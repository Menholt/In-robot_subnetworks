# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 10:23:28 2022

@author: lukas
"""

import numpy as np
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt

fs = 10000
new_fss = [5000, 1000, 500, 250, 100, 50, 10]
# new_fss = [250]



min_dist = 3
cell_radius = 2
velocity = 2
scenarios = [3]
realizations = 1024
t = 30



for scenario in tqdm(scenarios):
    hf = h5py.File('data\\scenario_{}\\interference_realization_{}_minDist_{}_cell_radius_{}_vel_{}_time_{}_fs_{}_scenario_{}.h5'.format(scenario, realizations, min_dist, cell_radius, velocity, t, fs, scenario), 'r')
    # hf = h5py.File('data\\performance_evaluation_scenarios\\interference_realization_{}_minDist_{}_cell_radius_{}_vel_{}_time_{}_fs_{}_size_60x60_nodes_16_scenario_{}.h5'.format(realizations, min_dist, cell_radius, velocity, t, fs, scenario), 'r')
    interference = np.array(hf.get('interference'))
    hf.close()
    
    # temp = interference
    # for i in range(len(interference)):
    #     if np.mean(10*np.log10(interference[len(interference)-i-1,:])) <= -150:
    #         temp = np.delete(temp, len(interference)-i-1, 0)
    # interference = temp
    # print(len(interference))
     
    for new_fs in new_fss:
        factor = int(fs/new_fs)        
        new_interference = np.zeros((realizations, new_fs*t))
        for i in range(realizations):
            new_interference[i] = interference[i,::factor]
        
    
        hf = h5py.File('data\\scenario_{}\\interference_realization_{}_minDist_{}_cell_radius_{}_vel_{}_time_{}_fs_{}_scenario_{}.h5'.format(scenario, realizations, min_dist, cell_radius, velocity, t, new_fs, scenario), 'w')
        # hf = h5py.File('data\\performance_evaluation_scenarios\\interference_realization_{}_minDist_{}_cell_radius_{}_vel_{}_time_{}_fs_{}_size_60x60_nodes_16_scenario_{}.h5'.format(realizations, min_dist, cell_radius, velocity, t, new_fs, scenario), 'w')
        hf.create_dataset('interference', data = new_interference)
        hf.close()
        

#%%
# scenario = 3
# velocity = 2
# for new_fs in new_fss:
#     factor = int(fs/new_fs)
#     hf = h5py.File('data\\scenario_{}\\interference_realization_{}_minDist_{}_cell_radius_{}_vel_{}_time_{}_fs_{}_scenario_{}.h5'.format(scenario, realizations, min_dist, cell_radius, velocity, t, new_fs, scenario), 'r')
#     interference = np.array(hf.get('interference'))
#     hf.close()
    
#     plt.plot(10*np.log10(interference[0, :int(10000/factor)]))
#     plt.title("scenario {} with fs: {}".format(scenario,new_fs))
#     plt.show()
    
