# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 10:23:28 2022

@author: lukas
"""
import numpy as np
import h5py
from tqdm import tqdm


vel = 2
min_dist = 3
cell_radius = 2
t = 30
realizations = 1024

fs = 10000
scenario = 5
# _nodes_32
new_interf = np.zeros((int(realizations), int(t*fs)))

for i in tqdm(range(16)):
    hf = h5py.File('data\\scenario_{}\\interference_realization_{}_minDist_{}_cell_radius_{}_time_{}_scenario_{}_v{}.h5'.format(scenario, realizations, min_dist, cell_radius, t, scenario, i), 'r')
    new_interf[int(i*64) : int((i+1)*64)] = np.array(hf.get('interference'))
    hf.close()

hf = h5py.File('data\\scenario_{}\\interference_realization_{}_minDist_{}_cell_radius_{}_time_{}_fs_{}_scenario_{}.h5'.format(scenario, realizations, min_dist, cell_radius, t, fs, scenario), 'w')
hf.create_dataset('interference', data = new_interf)
hf.close()

# for i in tqdm(range(16)):
#     hf = h5py.File('data\\norotate\\interference_realization_{}_minDist_{}_cell_radius_{}_vel_{}_time_{}_scenario_{}_kiss_no_rotate_v{}.h5'.format(realizations, min_dist, cell_radius, vel, t, scenario, i), 'r')
#     new_interf[int(i*64) : int((i+1)*64)] = np.array(hf.get('interference'))
#     hf.close()

# hf = h5py.File('data\\norotate\\interference_realization_{}_minDist_{}_cell_radius_{}_vel_{}_time_{}_fs_{}_scenario_{}_kiss_no_rotate.h5'.format(realizations, min_dist, cell_radius, vel, t, fs, scenario), 'w')
# hf.create_dataset('interference', data = new_interf)
# hf.close()