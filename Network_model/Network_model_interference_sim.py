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

def calc_inteference(in_put):
    np.random.seed()
    time, min_dist, cell_radius, vel, height, width, n_nodes, N_cg, alpha, delta, sigma, misalignment_on = in_put
    
    end_time = time
    f_c = 3e9  # carrier frequency
    c = 299792458 # speed of light
    fs = 10000 # samples frequency
    t_samples = int(end_time*fs) # number of samples
    t = np.linspace(0, end_time, t_samples, endpoint=False)
    time_step = t[1]
    kappa = 1 # gain
    
    # subnetwork parameters
    n_sensors = 1 # number of sensor acotuator pairs in each subnetwork
    active_devices = 1 # number of active sensor actuator pairs
    
    # shadow map
    stepsize = 1/20
    grf = i_f.createMap(width, height, sigma, delta, stepsize) # shadow map


    #Channel group
    channel_group = np.random.randint(0, N_cg, n_nodes)

    # small scale fading parameters
    M = 20 # number of waves
    M0 = int(M/4)
    alpha_n = np.array([2*np.pi*(n-0.5)/M for n in range(1,M0+1)])
    doppler = 2*np.pi*f_c*vel*np.cos(alpha_n)/c
    beta_n = [np.pi*n/(int(M0)) for n in range(1,M0+1)]
    theta = np.random.uniform(0, 2*np.pi, (np.sum(channel_group==channel_group[0])-1, active_devices+1, M0))
    # theta = np.random.uniform(0, 2*np.pi, (np.sum(channel_group==channel_group[0])-1, M0))
    
    node_loc = np.zeros((t_samples, n_nodes, 2), dtype=np.float32)
    
    # calculate the interference
    node_loc[0], counters = i_f.deployment(n_nodes, width, height, min_dist, cell_radius)
    directions = np.random.uniform(0, 2*np.pi, n_nodes)    
    
    sensor_cu_location = np.zeros((n_nodes, n_sensors+1, 2))
    
    for i in range(n_nodes):
        for j in range(n_sensors):
            sensor_cu_location[i, j+1] = i_f.interfere_displacement(cell_radius)
    
    activity = np.zeros((n_nodes, n_sensors*2,2))
    activity[:, :n_sensors, 0] = np.ones((n_nodes, n_sensors))
    
    if misalignment_on:
        misalignment = np.random.randint(0, 2*n_sensors, n_nodes)
    else:
        misalignment = np.zeros(n_nodes)
    
    for n in range(n_nodes):
        activity[n,:,0] = np.roll(activity[n,:,0], misalignment[n])
    
    
    direction_change_counter = np.zeros(n_nodes)
    interference = np.zeros((t_samples))+np.random.normal(0, 1e-10)**2
    # sensor_idxs = [[]for i in range(t_samples)]
    
    
    for i in range(t_samples-1):
        
        for n in range(n_nodes):
            # active_timeslots = np.random.choice(np.arange(n_sensors), active_devices, False)
            active_timeslots = np.arange(active_devices)
            for k in range(active_devices):
                activity[n, active_timeslots[k], 1] = k+1
                activity[n, active_timeslots[k] + n_sensors, 1] = k+1
            activity[n, :, 1] = np.roll(activity[n, :, 1], misalignment[n])
    
        node_loc[i+1], directions_new = i_f.random_direction(node_loc[i], directions, vel, min_dist, cell_radius, time_step, height, width)
        direction_change_counter += 1
        direction_change_counter = direction_change_counter * (directions != directions_new)
        directions = directions_new
        if np.max(direction_change_counter)>=1000:
            idx = np.argmax(direction_change_counter)
            node_loc[i+1, idx], directions[idx] = i_f.new_deployment(node_loc[i+1], idx, width, height, min_dist, cell_radius)

        # T_U_device = int((np.random.randint(0, n_sensors)+misalignment[0])%(2*n_sensors))
        T_U_device = int(misalignment[0])
        for j, cg_idx in enumerate(np.setdiff1d(np.where(channel_group == channel_group[0])[0],[0])):
            if activity[cg_idx,T_U_device,1]!=0:
                
                sensor_idx_or_0 = int(activity[cg_idx,T_U_device,0]*activity[cg_idx,T_U_device,1])
                # interfere_loc = i_f.interfere_location_no_rotate(node_loc[i+1, cg_idx], directions[cg_idx], sensor_cu_location[cg_idx, sensor_idx_or_0])
                interfere_loc = i_f.interfere_location(node_loc[i+1, cg_idx], directions[cg_idx], sensor_cu_location[cg_idx, sensor_idx_or_0])
                interference[i] += i_f.interference_power(kappa, node_loc[i+1, 0], interfere_loc, alpha, t[i], theta[j,sensor_idx_or_0,:], M0, doppler, beta_n, grf, delta, stepsize, height, width)
            
            # sensor_idxs[i].append(sensor_idx_or_0)

        activity[:,:,1] = np.zeros((n_nodes, n_sensors*2))
    return interference#, node_loc


if __name__== '__main__':
    height = 30
    width = 30
    min_dist = 3
    cell_radius = 2
    vel = 2
    n_nodes = 16 # number of subnetworks
    N_cg = 4 # number of channel groups
    alpha = 3 # pathloss gain
    delta = 5 # decorrelation distance
    sigma = 3 # variance
    misalignment_on = True
    t1 = time.time()
    N = int(mp.cpu_count())
    t = 30
    in_put = []
    for i in range(N):
        in_put.append((time, min_dist, cell_radius, vel, height, width, n_nodes, N_cg, alpha, delta, sigma, misalignment_on))
    in_put = tuple(in_put)
    realizations = 16
    # interference = np.zeros((N*realizations, t*10000))
    for i in tqdm(range(realizations)):
        # in_put = tuple(map(tuple, in_put))
        pool = mp.Pool(processes=N)
        result = pool.map_async(calc_inteference, in_put)
        pool.close()
        pool.join()

        # interference[i*N:(i+1)*N] = result.get()
        interference = result.get()
        hf = h5py.File('data/interference_realization_{}_minDist_{}_cell_radius_{}_vel_{}_time_{}_scenario_3_v{}.h5'.format(N*realizations, min_dist, cell_radius, vel, t, i), 'w')
        hf.create_dataset('interference', data = interference)
        hf.close()
        
