# -*- coding: utf-8 -*-
"""
Created on Tue May  3 10:58:51 2022

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
    time, min_dist, cell_radius, vel = in_put
    
    height = 30
    width = 30
    end_time = time
    f_c = 3e9  # carrier frequency
    c = 299792458 # speed of light
    fs = 10000 # samples frequency
    t_samples = int(end_time*fs) # number of samples
    t = np.linspace(0, end_time, t_samples, endpoint=False)
    time_step = t[1]
    
    # subnetwork parameters
    # cell_radius = 2
    # min_dist = 3
    n_nodes = 16 # number of subnetworks
    n_sensors = 18 # number of sensor acotuator pairs in each subnetwork
    # vel = 20 # velocity
    active_devices = 18 # number of active sensor actuator pairs
    
    # shadow map parameter
    delta = 5 # decorrelation distance
    sigma = 3 # variance
    stepsize = 1/20
    grf = i_f.createMap(width, height, sigma, delta, stepsize) # shadow map


    #Channel group parameter
    N_cg = 4 # number of channel groups
    # channel_group = np.random.randint(0, N_cg, n_nodes)
    channel_group = np.zeros(n_nodes)
    channel_group[0:2] = [1,1]

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
    
    
    misalignment = np.random.randint(0, 2*n_sensors, n_nodes)
    # misalignment = np.zeros(n_nodes)
    
    for n in range(n_nodes):
        activity[n,:,0] = np.roll(activity[n,:,0], misalignment[n])
    
    
    direction_change_counter = np.zeros(n_nodes)
    interference = np.zeros((t_samples))+np.random.normal(0, 1e-10)**2
    path_l = np.zeros((t_samples))
    ss_fading = np.zeros((t_samples))
    shadow = np.zeros((t_samples))
    # sensor_idxs = [[]for i in range(t_samples)]
    
    
    for i in range(t_samples-1):
        
        for n in range(n_nodes):
            active_timeslots = np.random.choice(np.arange(n_sensors), active_devices, False)
            # active_timeslots = np.arange(active_devices)
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

        T_U_device = int((np.random.randint(0, n_sensors)+misalignment[0])%(2*n_sensors))
        # T_U_device = 0
        h_var = np.zeros(4)
        for j, cg_idx in enumerate(np.setdiff1d(np.where(channel_group == channel_group[0])[0],[0])):
            if activity[cg_idx,T_U_device,1]!=0:
                sensor_idx_or_0 = int(activity[cg_idx,T_U_device,0]*activity[cg_idx,T_U_device,1])
                interfere_loc = i_f.interfere_location_2(node_loc[i+1, cg_idx], directions[cg_idx], sensor_cu_location[cg_idx, sensor_idx_or_0])
                
                h_var += i_f.interference_power_med_resten(kappa, node_loc[i+1, 0], interfere_loc, alpha, t[i], theta[j,sensor_idx_or_0,:], M0, doppler, beta_n, grf, delta, stepsize, height, width)
                interference[i] += h_var[0]
                path_l[i] += h_var[1]
                ss_fading[i] += h_var[2]
                shadow[i] += h_var[3]
            
            # sensor_idxs[i].append(sensor_idx_or_0)

        activity[:,:,1] = np.zeros((n_nodes, n_sensors*2))
    return interference, path_l, ss_fading, shadow

# if __name__== '__main__':
#     min_dist = 3
#     cell_radius = 2

#     t1 = time.time()
#     N = int(mp.cpu_count())
#     t = 30
#     in_put = []
#     for i in range(N):
#         in_put.append((t, min_dist, cell_radius))
#     in_put = tuple(in_put)
#     realizations = 16
#     interference = np.zeros((N, t*10000))
#     node_loc = np.zeros((N, t*10000, 16, 2), dtype=np.float32)
#     for i in tqdm(range(realizations)):
#         # in_put = tuple(map(tuple, in_put))
#         pool = mp.Pool(processes=N)
#         result = pool.map_async(calc_inteference, in_put)
#         pool.close()
#         pool.join()

#         a = result.get()
#         for j in range(N):
#             interference[j] = a[j][0]
#             node_loc[j] = a[j][1]
        
#         hf = h5py.File('data/interference_nokiss_realization_{}_minDist_{}_cell_radius_{}_time_{}_scenario_6_v{}.h5'.format(N, min_dist, cell_radius, t, i), 'w')
#         hf.create_dataset('interference', data = interference)
#         hf.close()
        
#         hf = h5py.File('data/node_location_nokiss_realization_{}_minDist_{}_cell_radius_{}_time_{}_v{}.h5'.format(N, min_dist, cell_radius, t, i), 'w')
#         hf.create_dataset('node_loc', data = node_loc)
#         hf.close()




if __name__== '__main__':
    min_dist = 3
    cell_radius = 2
    vel = 2
    t1 = time.time()
    N = int(mp.cpu_count())
    time = 1
    
    in_put =(time, min_dist, cell_radius, vel)

    interference, path_l, ss_fading, shadow = calc_inteference(in_put)
    
    #%%
    fs = 10000
    
    fig, axs = plt.subplots(4, 1, sharex='all')
    fig.set_figheight(6*4)
    fig.set_figwidth(18)
    t = np.arange(time*fs)/fs
    font = 25
    axs[0].plot(t[:2000], 10*np.log10(ss_fading[:2000]))
    axs[0].set_ylabel("SS-fading [dB]", fontsize = font)
    axs[0].tick_params(labelsize = font)
    
    axs[1].plot(t[:2000], 10*np.log10(path_l[:2000]))
    axs[1].set_ylabel("Path-loss [dB]", fontsize = font)
    axs[1].tick_params(labelsize = font)
    
    axs[2].plot(t[:2000], 10*np.log10(shadow[:2000]))
    axs[2].set_ylabel("Shadowing [dB]", fontsize = font)
    axs[2].tick_params(labelsize = font)
    
    axs[3].plot(t[:2000], 10*np.log10(interference[:2000]))
    axs[3].set_ylabel("Interference power [dB]", fontsize = font)
    axs[3].tick_params(labelsize = font)

    plt.xticks(fontsize = font)
    plt.xlabel("Time [s]", fontsize = font)
    plt.tight_layout()
    plt.savefig("figures/interference_plus_resten.pdf", bbox_inches = "tight")
    plt.show()
    
    #%%
    t = np.arange(time*fs)/fs
    font = 25
    plt.figure(figsize=(18,6))
    plt.plot(t[:2000], 10*np.log10(ss_fading[:2000]))
    plt.ylabel("SS-fading [dB]", fontsize = font)
    plt.xlabel("Time [s]", fontsize = font)
    plt.yticks(fontsize = font)
    plt.xticks(fontsize = font)
    plt.tight_layout()
    plt.savefig("figures/ss_fading.pdf")
    plt.show()