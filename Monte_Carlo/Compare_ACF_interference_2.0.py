# -*- coding: utf-8 -*-
"""
Created on Mon May 30 09:02:16 2022

@author: EG
"""

import numpy as np
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl


min_dist = 3
cell_radius = 2
vel = 2
fs = 10000
t = 30
scenarios = [1, 2, 3, 4]


N_samples = 100
int_samples = 100
n_iter = 16


#%% Scenario 1,2,4
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'purple','k']

x_acf = np.linspace(0,99,1000)

hf_MC = h5py.File('data/monte_acfs_iter_{}_Nsamples_{}_intsamples_{}.h5'.format(n_iter, N_samples, int_samples), 'r')

taus = np.array(hf_MC.get('taus'))

font = 25
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
fig.tight_layout=True
scenarios = [1,2,3,4,5]
for i in range(len(scenarios)):
    scenario = scenarios[i]
    hf = h5py.File('../Network_model/data/scenario_{}/acf_minDist_{}_cell_radius_{}_vel_{}_fs_{}_scenario_{}.h5'.format(scenario, min_dist, cell_radius, vel, fs,scenario), 'r')
    acfs = np.array(hf.get('acf'))
    hf.close()

    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    quan = np.quantile(acfs, quantiles, axis=0)
    for j in range(len(quantiles)):
        axs[i].plot(x_acf[:500], quan[j,:500], color = colors[j])
        
    acf_MC = np.array(hf_MC.get('acf_{}'.format(scenarios[i])))

    axs[i].plot(taus*1000, acf_MC, color = colors[5])

    axs[i].set_title("ACF of scenario {}".format(scenario), fontsize = font)

    # axs[i].legend(quantiles+["MC ACF"], fontsize = font - 5, title = "Quantiles", title_fontsize = font-5)
    axs[i].tick_params(labelsize = font - 5)
    axs[i].grid()

plt.legend(quantiles+["MC ACF"], fontsize = font - 5, title = "Quantiles", title_fontsize = font-5, loc = [1.1,0.285])
# fig.text(0.51, 0.00, labels[0], fontsize = font, ha='center')

axs[-1].set_xlabel('Lag [ms]',fontsize = font)
axs[2].set_ylabel('ACF', fontsize = font)
plt.savefig('figures\\acf_quantiles_scenarios.pdf', bbox_inches='tight')
plt.show()

hf_MC.close()
