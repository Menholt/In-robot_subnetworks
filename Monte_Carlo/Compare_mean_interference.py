# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 09:14:28 2022

@author: lukas
"""
import numpy as np
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl


def MC_mean_compare(scenario, n_iter, N_samples, L_samples, realizations, min_dist, cell_radius, velocity, t, fs):
    
    # MC mean
    int_means_MC = np.zeros((64, 16))
    if scenario<=2:    
        hf = h5py.File('data/monte_mean_sensor_iter_{}_Nsamples_{}_Lsamples_{}.h5'.format(n_iter, N_samples, L_samples), 'r')
    else:
        hf = h5py.File('data/monte_mean_iter_{}_Nsamples_{}_Lsamples_{}.h5'.format(n_iter, N_samples, L_samples), 'r')
    for i in range(64):
        int_means_MC[i] = np.array(hf.get('monte_means_{}'.format(i)))
    hf.close()
    
    if scenario == 2 or scenario == 4 or scenario == 5:
        int_means_MC = np.array(int_means_MC).reshape(-1)
        mean_MC = np.mean(int_means_MC*5)
        std = np.std(int_means_MC*5)
    else:
        int_means_MC = np.array(int_means_MC).reshape(-1)
        mean_MC = np.mean(int_means_MC)
        std = np.std(int_means_MC)

    lower = 10*np.log10(mean_MC-1.96*std)
    upper = 10*np.log10(mean_MC+1.96*std)
    
    # Interference model mean
    hf = h5py.File('../Network_model/data/scenario_{}/interference_realization_{}_minDist_{}_cell_radius_{}_vel_{}_time_{}_fs_{}_scenario_{}.h5'.format(scenario, realizations, min_dist, cell_radius, velocity, t, fs, scenario), 'r')
    interference = np.array(hf.get('interference'))
    hf.close()
    
    temp = interference
    for i in range(len(interference)):
        if np.mean(10*np.log10(interference[len(interference)-i-1,:])) <= -150:
            temp = np.delete(temp, len(interference)-i-1, 0)
    interference = temp
    
    # plot
    quantiles = [0.025,0.25,0.5,0.75,0.975]

    
    means = 10*np.log10(np.mean(interference, axis = 1))
    quans = np.quantile(means, quantiles)

    return lower, upper, means, quans     



min_dist = 3
cell_radius = 2

t = 30
realizations = 1024
scenarios = [1,2,3,4,5]
fs = 10000
velocity = 2

n_iter = 1024

N_samples = 50
L_samples = 10
N = 64


colours = ["r", "orange", "lime", "orange", "r"]

font = 25
width = 3

fig = plt.figure()
fig.set_figheight(18)
fig.set_figwidth(18)
spec = mpl.gridspec.GridSpec(ncols=4, nrows=3) # 6 columns evenly divides both 2 & 3

ax1 = fig.add_subplot(spec[0,0:2]) # row 0 with axes spanning 2 cols on evens
ax2 = fig.add_subplot(spec[0,2:4])
ax3 = fig.add_subplot(spec[1,0:2])
ax4 = fig.add_subplot(spec[1,2:4]) # row 0 with axes spanning 2 cols on odds
ax5 = fig.add_subplot(spec[2,1:3])
axs = np.array([ax1, ax2, ax3, ax4, ax5])

for i, scenario in enumerate(scenarios):
    MC_lower, MC_upper, means, quans = MC_mean_compare(scenario, n_iter, N_samples, L_samples, realizations, min_dist, cell_radius, velocity, t, fs)
    a ,_ ,__ = axs[i].hist(means, bins=32, density=True)
    for j in range(len(quans)):
        axs[i].vlines(quans[j], 0, max(a), colours[j], linewidth=width)
    axs[i].vlines(MC_lower, 0, max(a), 'k', linewidth=width, linestyle="dashed")
    axs[i].vlines(MC_upper, 0, max(a), 'k', linewidth=width, linestyle="dashed")
    axs[i].set_title("Scenario {}".format(scenario), fontsize = font)
    axs[i].tick_params(labelsize = font)
    
axs[2].set_ylabel("Normalized frequency", fontsize = font)
axs[i].legend(("2.5%", "25%", "50%", "75%", "97.5%", "MC int"), fontsize = font-5, loc=[1.1,0.375])  
axs[i].set_xlabel("Means", fontsize = font)
# plt.tight_layout()

plt.savefig("figures/histogram_mc_means_scenarios.pdf", bbox_inches = "tight")
plt.show()
#%% Comparison

scenario = 5

plt.figure(figsize=(18,6))
MC_lower, MC_upper, means, quans = MC_mean_compare(scenario, n_iter, N_samples, L_samples, realizations, min_dist, cell_radius, velocity, t, fs)
a ,_ ,__ = plt.hist(means, bins=32, density=True)
for j in range(len(quans)):
    plt.vlines(quans[j], 0, max(a), colours[j], linewidth=width)
plt.vlines(MC_lower, 0, max(a), 'k', linewidth=width, linestyle="dashed")
plt.vlines(MC_upper, 0, max(a), 'k', linewidth=width, linestyle="dashed")
plt.title("Scenario {}".format(scenario), fontsize = font)
plt.ylabel("Normalized frequency", fontsize = font)
plt.xlabel("Means", fontsize = font)
plt.legend(("2.5%", "25%", "50%", "75%", "97.5%", "MC"), fontsize = font-5)
plt.xticks(fontsize = font)
plt.yticks(fontsize = font)

# plt.tight_layout()

plt.savefig("figures/histogram_mc_means_scenario_5.pdf", bbox_inches = "tight")
plt.show()







# mean_int_MC = 10*np.log10(np.mean(int_means_MC))
# # mean_int = 10*np.log10(np.mean(interferences))

# # print("Mean interference model: {}".format(mean_int))
# print("Mean interference MC: {}".format(mean_int_MC))

