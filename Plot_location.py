# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 08:51:33 2022

@author: EG
"""
import matplotlib.pyplot as plt
import interference_functions as i_f
import numpy as np
from tqdm import tqdm

def plot_subnetwork_locations(locations, row, col, title, channel_group, colours, cell_radius, min_dist, n_nodes):    
    inner_cir = plt.Circle((locations[0,0], locations[0,1]), min_dist/2, color=colours[channel_group[0]], fill=False)
    outer_cir = plt.Circle((locations[0,0], locations[0,1]), cell_radius, color=colours[channel_group[0]], fill=False)
    axs[row, col].add_patch(inner_cir)
    axs[row, col].add_patch(outer_cir)
    axs[row, col].set_title(title, fontsize = 45)
    for n in range(1,n_nodes):
        inner_cir = plt.Circle((locations[n,0], locations[n,1]), min_dist/2, color=colours[channel_group[n]], fill=False)
        outer_cir = plt.Circle((locations[n,0], locations[n,1]), cell_radius, color=colours[channel_group[n]], fill=False)
        axs[row, col].add_patch(inner_cir)
        axs[row, col].add_patch(outer_cir)

def plot_subnetwork_locations_single(locations, channel_group, colours, interfere_disp, cell_radius, min_dist, n_nodes):
    for n in range(n_nodes):
        inner_cir = plt.Circle((locations[n,0], locations[n,1]), min_dist/2, color=colours[channel_group[n]], fill=False)
        outer_cir = plt.Circle((locations[n,0], locations[n,1]), cell_radius, color=colours[channel_group[n]], fill=False)
        axs.plot(locations[n,0], locations[n,1], 'v', color='b', markersize=12)

        for i in range(n_sensors):
            intf_loc = i_f.interfere_location(node_loc[n], directions[n], interfere_disp[n, i])
            axs.plot(intf_loc[0], intf_loc[1], 'v', color='r', markersize=8)
        axs.add_patch(inner_cir)
        axs.add_patch(outer_cir)

# =============================================================================
# #plot from page 23 of the project 
# =============================================================================
if __name__== '__main__':
    #plot deployments and mobility    
    height = 30
    width = 30
    cell_radius = 2
    min_dist = 3  
    vel = 2
    n_nodes = 16
    
    time = 5.0001
    fs = 10000
    t_samples = int(time*fs)
    t = np.linspace(0, time, t_samples)
    time_step = t[1]
    
    node_loc, counters = i_f.deployment(n_nodes, width, height, min_dist, cell_radius)    
    directions = np.random.uniform(0, 2*np.pi, n_nodes)
    
    N_cg = 4 # number of channel groups
    channel_group = np.random.randint(0, N_cg, n_nodes)
    colours = ["b", "g", "r", "m"]
    
    font = 45
    fig, axs = plt.subplots(2, 3, sharex='all', sharey='all')
    fig.set_figheight(9*2)
    fig.set_figwidth(9*3)
    axs[0, 0].set_xlim((-15, 15))
    axs[0, 0].set_ylim((-15, 15))
    
    for i in range(2):
        for j in range(3):
            axs[i,j].tick_params(labelsize = font - 10)
    
    for i in tqdm(range(t_samples)):
        node_loc, directions = i_f.random_direction(node_loc, directions, vel, min_dist, cell_radius, time_step, height, width) 
        if (i == 0):
            axs[0, 0] = plot_subnetwork_locations(node_loc, 0, 0, r't=0 [s]', channel_group, colours, cell_radius, min_dist, n_nodes)
        if (i == 10000):
            axs[0, 1] = plot_subnetwork_locations(node_loc, 0, 1, r't=1 [s]', channel_group, colours, cell_radius, min_dist, n_nodes)
        if (i == 20000):
            axs[0, 2] = plot_subnetwork_locations(node_loc, 0, 2, r't=2 [s]', channel_group, colours, cell_radius, min_dist, n_nodes)
        if (i == 30000):
            axs[1, 0] = plot_subnetwork_locations(node_loc, 1, 0, r't=3 [s]', channel_group, colours, cell_radius, min_dist, n_nodes)
        if (i == 40000):
            axs[1, 1] = plot_subnetwork_locations(node_loc, 1, 1, r't=4 [s]', channel_group, colours, cell_radius, min_dist, n_nodes)
        if (i == 50000):
            axs[1, 2] = plot_subnetwork_locations(node_loc, 1, 2, r't=5 [s]', channel_group, colours, cell_radius, min_dist, n_nodes)
    plt.tight_layout=True
    # plt.savefig("figures\\deployment_and_mobility.pdf")
    plt.show()
    

# =============================================================================
# #plot from the presentation can be changed to be the plot from the front page 
# #by changing the colors and setting min_dist = 2*cell_radius
# =============================================================================
if __name__=='__main__':
    #plot deployments and mobility    
    height = 30
    width = 40
    cell_radius = 2
    min_dist = 3
    vel = 2
    n_nodes = 16
    n_sensors = 3
    
    time = 0.001
    fs = 10000
    t_samples = int(time*fs)
    t = np.linspace(0, time, t_samples)
    time_step = t[1]
    
    node_loc, counters = i_f.deployment(n_nodes, width, height, min_dist, cell_radius)    
    directions = np.random.uniform(0, 2*np.pi, n_nodes)
    
    N_cg = 4 # number of channel groups
    channel_group = np.random.randint(0, N_cg, n_nodes)
    colours = ["b", "g", "r", "m"]
    # colours = ["k", "k", "k", "k"]
    
    font = 45
    fig, axs = plt.subplots(1, 1, sharex='all', sharey='all')
    fig.set_figheight(12)
    fig.set_figwidth(16)
    axs.set_xlim((-20, 20))
    axs.set_ylim((-15, 15))
    
    axs.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    
    interfere_disp = np.zeros((n_nodes, n_sensors,2))
    for i in range(n_nodes):
        for j in range(n_sensors):
            interfere_disp[i, j] = i_f.interfere_displacement(cell_radius)
    for i in tqdm(range(t_samples)):
        node_loc, directions = i_f.random_direction(node_loc, directions, vel, min_dist, cell_radius, time_step, height, width) 
        if (i == 0):
            axs = plot_subnetwork_locations_single(node_loc, channel_group, colours, interfere_disp, cell_radius, min_dist, n_nodes)
    plt.tight_layout=True
    # plt.savefig("figures\\front_page_beamer.pdf")
    plt.show()