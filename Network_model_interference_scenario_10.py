# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 10:29:35 2022

@author: EG
"""
# -*- coding: utf-8 -*-

import numpy as np
import h5py
import multiprocessing as mp
from tqdm import tqdm
import time
import interference_functions as i_f
import matplotlib.pyplot as plt

def calc_inteference(in_put):
    np.random.seed()
    time, min_dist, cell_radius = in_put
    
    height = 30
    width = 30
    end_time = time
    f_c = 3e9  # carrier frequency
    c = 299792458 # speed of light
    fs = 10000 # samples frequency
    t_samples = int(end_time*fs) # number of samples
    t = np.linspace(0, end_time, t_samples)
    time_step = t[1]
    
    # subnetwork parameters
    # cell_radius = 2.5
    # min_dist = 2.5
    n_nodes = 16 # number of subnetworks
    n_sensors = 1 # number of sensor acotuator pairs in each subnetwork
    vel = 2 # velocity
    active_devices = 1 # number of active sensor actuator pairs
    
    # shadow map parameter
    delta = 5 # decorrelation distance
    sigma = 3 # variance
    stepsize = 1/20
    grf = i_f.createMap(width, height, sigma, delta, stepsize) # shadow map


    #Channel group parameter
    N_cg = 1 # number of channel groups
    channel_group = np.random.randint(0, N_cg, n_nodes)
    channel_group = np.zeros(n_nodes)
    # channel_group[0:2] = [1,1]

    # interference power parameters
    kappa = 1 # gain
    alpha = 3 # pathloss gain
    # small scale fading parameters
    M = 20 # number of waves
    M0 = int(M/4)
    alpha_n = np.array([2*np.pi*(n-0.5)/M for n in range(1,M0+1)])
    doppler = 2*np.pi*f_c*vel*np.cos(alpha_n)/c
    beta_n = [np.pi*n/(int(M0)) for n in range(1,M0+1)]
    theta = np.random.uniform(0, 2*np.pi, (np.sum(channel_group==channel_group[0])-1, active_devices+1, M0))
    # theta = np.random.uniform(0, 2*np.pi, (np.sum(channel_group==channel_group[0])-1, M0))
    
    # calculate the interference
    node_loc, counters = i_f.deployment(n_nodes, width, height, min_dist, cell_radius)
    directions = np.random.uniform(0, 2*np.pi, n_nodes)    
    
    sensor_cu_location = np.zeros((n_nodes, n_sensors+1, 2))
    
    for i in range(n_nodes):
        for j in range(n_sensors):
            sensor_cu_location[i, j+1] = i_f.interfere_displacement(cell_radius)
        
    
    misalignment = np.zeros((2, n_nodes))
    misalignment[0,:] = np.random.randint(0, 5, n_nodes)/4
    misalignment[1,:] = 1-misalignment[0,:]
    
    
    
    direction_change_counter = np.zeros(n_nodes)
    interference = np.zeros((t_samples))+1e-20
    # sensor_idxs = [[]for i in range(t_samples)]
        
    # pbar = tqdm(range(t_samples), ascii = '.'+'>'*8+'=')    #progress bar
    # for p, i in zip(pbar, range(t_samples)):
    for i in range(t_samples):
    
        node_loc, directions_new = i_f.random_direction(node_loc, directions, vel, min_dist, cell_radius, time_step, height, width)
        direction_change_counter += 1
        direction_change_counter = direction_change_counter * (directions != directions_new)
        directions = directions_new
        if np.max(direction_change_counter)==1000:
            idx = np.argmax(direction_change_counter)
            node_loc[idx], directions[idx] = i_f.new_deployment(node_loc, idx, width, height, min_dist, cell_radius)

        for j, cg_idx in enumerate(np.setdiff1d(np.where(channel_group == channel_group[0])[0],[0])):
            for m in range(2):
                interfere_loc = i_f.interfere_location(node_loc[cg_idx], directions[cg_idx], sensor_cu_location[cg_idx, m])
                interference[i] += misalignment[m, cg_idx]*i_f.interference_power(kappa, node_loc[0], interfere_loc, alpha, t[i], theta[j,m,:], M0, doppler, beta_n, grf, delta, stepsize, height, width)

    return interference

if __name__== '__main__':
    min_dist = 3
    cell_radius = 2

    t1 = time.time()
    N = int(mp.cpu_count())
    t = 30
    in_put = []
    for i in range(N):
        in_put.append((t, min_dist, cell_radius))
    in_put = tuple(in_put)
    realizations = 64
    interference = np.zeros((N*realizations, t*10000))
    for i in tqdm(range(realizations)):
        # in_put = tuple(map(tuple, in_put))
        pool = mp.Pool(processes=N)
        result = pool.map_async(calc_inteference, in_put)
        pool.close()
        pool.join()
    
        # interference[i*N:(i+1)*N] = result.get()
        interference = result.get()
        
        hf = h5py.File('data/interference_realization_{}_minDist_{}_cell_radius_{}_time_{}_scenario_5_v{}.h5'.format(N*realizations, min_dist, cell_radius, t, i), 'w')
        hf.create_dataset('interference', data = interference)
        hf.close()

# if __name__ == "__main__":
#     t = 0.1
#     t1 = time.time()
#     fs = 10000

#     interf = calc_inteference(t)
#     x = np.linspace(0, t, int(t*fs), endpoint=False)
#     font = 25
#     plt.figure(figsize=(12,6))
#     plt.plot(x, 10*np.log10(interf[:]))
#     plt.xlabel("Time [s]", fontsize = font)
#     plt.ylabel("Interference power [dB]", fontsize = font)
#     plt.xticks(fontsize = font-5)
#     plt.yticks(fontsize = font-5)
#     plt.title("Inteference power", fontsize =font)
#     # plt.savefig("Interference_power.pdf")
#     # plt.plot(10*np.log10(interf[:,2]))

#     print((time.time()-t1))

