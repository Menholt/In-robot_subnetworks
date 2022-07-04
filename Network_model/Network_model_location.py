# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 12:36:15 2021

@author: EG
"""
import numpy as np
import h5py
import multiprocessing as mp
from tqdm import tqdm
import time
import interference_functions as i_f
import matplotlib.pyplot as plt

def calc_inteference(time):
    np.random.seed()
        
    height = 30
    width = 30
    end_time = time
    fs = 10000 # samples frequency
    t_samples = int(end_time*fs) # number of samples
    t = np.linspace(0, end_time, t_samples)
    time_step = t[1]
    
    # subnetwork parameters
    cell_radius = 2.5
    min_dist = 2
    n_nodes = 16 # number of subnetworks
    vel = 2 # velocity

    node_loc = np.zeros((t_samples, n_nodes, 2), dtype=np.float32)
    # Find locations
    node_loc[0], counters = i_f.deployment(n_nodes, width, height, min_dist, cell_radius)
    directions = np.random.uniform(0, 2*np.pi, n_nodes)    
    direction_change_counter = np.zeros(n_nodes)
    for i in range(t_samples-1):
        node_loc[i+1], directions_new = i_f.random_direction(node_loc[i], directions, vel, min_dist, cell_radius, time_step, height, width)
        direction_change_counter += 1
        direction_change_counter = direction_change_counter * (directions != directions_new)
        directions = directions_new
        if np.max(direction_change_counter)>=10:
            idx = np.argmax(direction_change_counter)
            node_loc[i+1, idx], directions[idx] = i_f.new_deployment(node_loc[i+1], idx, width, height, min_dist, cell_radius)

    return node_loc

if __name__== '__main__':
    min_dist = 2.5
    cell_radius = 2.5

    t1 = time.time()
    N = int(mp.cpu_count())
    t = 30
    in_put = tuple(np.ones(N)*t)
    realizations = 16
    
    for i in tqdm(range(realizations)):
        # node_loc = np.zeros((N, t*10000, 16, 2), dtype=np.float32)
        # in_put = tuple(map(tuple, in_put))
        pool = mp.Pool(processes=N)
        result = pool.map_async(calc_inteference, in_put)
        pool.close()
        pool.join()

        node_loc = result.get()
        
        hf = h5py.File('data/node_location_realization_{}_minDist_{}_cell_radius_{}_time_{}_v{}.h5'.format(N, min_dist, cell_radius, t, i), 'w')
        hf.create_dataset('node_loc', data = node_loc)
        hf.close()
        
        # x = np.linspace(0, 30, 61)
        # hist2d = np.histogram2d(node_loc.reshape(64*10000*16,2)[:,0],node_loc.reshape(64*10000*16,2)[:,1], bins = x)
        # hf = h5py.File('data/node_location_hist_realization_{}_minDist_{}_cell_radius_{}_time_{}_v{}.h5'.format(N, min_dist, cell_radius, t, i), 'w')
        # hf.create_dataset('hist2d', data = hist2d)
        # hf.close()
