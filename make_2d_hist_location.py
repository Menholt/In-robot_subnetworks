# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 16:00:57 2022

@author: EG
"""

import numpy as np
import h5py
import multiprocessing as mp
from tqdm import tqdm
import matplotlib.pyplot as plt


min_dist = 2.5
cell_radius = 2.5

N = int(mp.cpu_count())
t = 30
in_put = tuple(np.ones(N)*t)
realizations = 8
hist2d_all = np.zeros((60, 60))
for i in tqdm(range(realizations)):
    
    hf = h5py.File('data/node_location_realization_{}_minDist_{}_cell_radius_{}_time_{}_v{}.h5'.format(64, min_dist, cell_radius, t, i), 'r')
    node_loc = np.array(hf.get('node_loc'))
    hf.close()

    
    x = np.linspace(2.5, 27.5, 61) -15
    hist2d = np.histogram2d(node_loc.reshape(64*t*10000*16,2)[:,0],node_loc.reshape(64*t*10000*16,2)[:,1], bins = x, density = True)
    hist2d_all += hist2d[0]
    
    # plt.pcolormesh(hist2d[2], hist2d[1], hist2d[0])
    # plt.ylabel("Length from peak to valley in samples")
    # plt.xlabel('Difference from peaks to valley in dB')
    # plt.title("{}".format(i))
    # plt.colorbar()
    # plt.show()

hf = h5py.File('data/node_location_hist_realization_{}_minDist_{}_cell_radius_{}_time_{}.h5'.format(N, min_dist, cell_radius, t), 'w')
hf.create_dataset('hist2d', data = hist2d_all)
hf.close()

plt.pcolormesh(hist2d[2], hist2d[1], hist2d_all)
plt.ylabel("Length from peak to valley in samples")
plt.xlabel('Difference from peaks to valley in dB')
plt.colorbar()
plt.show()


hf = h5py.File('data/node_location_realization_{}_minDist_{}_cell_radius_{}_time_{}_v{}.h5'.format(64, min_dist, cell_radius, t, 15), 'r')
node_loc = np.array(hf.get('node_loc'))
hf.close()





