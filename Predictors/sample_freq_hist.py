# -*- coding: utf-8 -*-
"""
Created on Mon May  9 10:20:33 2022

@author: lukas
"""

import numpy as np
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt

fss = [5000, 1000, 500, 250]
# new_fss = [10000]



min_dist = 3
cell_radius = 2
velocity = 2
scenario = 3
realizations = 1024
t = 30

bins = np.linspace(0, 5, 31)


fig, axs = plt.subplots(2, 2, sharex=True)
fig.set_figheight(12)
fig.set_figwidth(18)
plt.tight_layout=True

font = 25
width = 3

for i,fs in enumerate(fss):
    hf = h5py.File('..\\Network_model\\data\\scenario_{}\\interference_realization_{}_minDist_{}_cell_radius_{}_vel_{}_time_{}_fs_{}_scenario_{}.h5'.format(scenario, realizations, min_dist, cell_radius, velocity, t, fs, scenario), 'r')
    interference = np.array(hf.get('interference'))[:, :-1]
    hf.close()
    
    temp = interference
    for j in range(len(interference)):
        if np.mean(10*np.log10(interference[len(interference)-j-1,:])) <= -150:
            temp = np.delete(temp, len(interference)-j-1, 0)
    interference = temp
    
    diffs = np.abs(np.diff(10*np.log10(interference), axis = 1))
    
    mid = np.median(diffs)
    
    a,_, __ = axs[int(i//2),int(i%2)].hist(diffs.reshape(-1), bins = bins, density = True)
    axs[int(i//2),int(i%2)].vlines(mid, 0, np.max(a), "k", linewidth = width)
    axs[int(i//2),int(i%2)].set_title("scenario {} with fs: {} [Hz]".format(scenario, fs), fontsize = font)
    # if int(i%2) == 0:
        # axs[int(i//2),int(i%2)].set_ylabel("Normalized frequency", fontsize = font)
    # if  i >= 2:
        # axs[int(i//2),int(i%2)].set_xlabel("Difference [dB]", fontsize = font)
    axs[int(i//2),int(i%2)].tick_params(labelsize = font)
    # axs[int(i//2),int(i%2)].legend(("Median = {:.3F}".format(mid), "Differences"), fontsize = font-5)
    axs[int(i//2),int(i%2)].legend((["Median = {:.3F}".format(mid)]), fontsize = font-5)

fig.text(0.51, 0.05, "Difference [dB]", fontsize = font, ha='center')
fig.text(0.05,0.35, 'Normalized frequency', fontsize = font, ha='center', rotation='vertical')

plt.savefig("figures/histogram_diffs.pdf", bbox_inches = "tight")        
plt.show()