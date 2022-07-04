# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 11:55:34 2022

@author: EG
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy import ndimage, signal
from tqdm import tqdm
from scipy.io import savemat

realizations = 1024
min_dist = 3
cell_radius = 2
vel = 2
fs = 10000
t = 30
scenarios = [1, 2, 3, 4, 5]
plotting_time = 0.5


font = 25
fig, axs = plt.subplots(3, 2, sharex=True)
fig.set_figheight(3*6)
fig.set_figwidth(16)
plt.tight_layout=True
x = np.linspace(0, plotting_time, int(plotting_time*fs))
for j, scenario in enumerate(scenarios):
    i = j + 1
    hf = h5py.File('data\\scenario_{}\\interference_realization_{}_minDist_{}_cell_radius_{}_vel_{}_time_{}_fs_{}_scenario_{}.h5'.format(scenario, realizations, min_dist, cell_radius, vel, t, fs, scenario), 'r')
    interferences = np.array(hf.get('interference'))[0, :int(fs*plotting_time)]
    hf.close()

    axs[int(i//2), int(i%2)].plot(x, 10*np.log10(interferences))
    # axs[int(i//2), int(i%2)].set_ylabel('Power [dB]',fontsize = font)
    axs[int(i//2), int(i%2)].set_title('Scenario {}'.format(scenario), fontsize = font)
    axs[int(i//2), int(i%2)].tick_params(labelsize = font)
    axs[int(i//2), int(i%2)].grid()
    # axs[int(i//2), int(i%2)].set_xlabel("Time [s]", fontsize = font)

axs[1, 0].set_ylabel('Power [dB]',fontsize = font)
fig.text(0.51, 0.081, "Time [s]", fontsize = font, ha='center')

scenario = 2
hf = h5py.File('data\\interference_realization_{}_minDist_{}_cell_radius_{}_vel_{}_time_{}_scenario_{}.h5'.format(realizations, min_dist, cell_radius, vel, t, scenario), 'r')
interferences = np.array(hf.get('interference'))[0, :int(fs*plotting_time)]
hf.close()

axs[0, 0].plot(x, 10*np.log10(interferences))
# axs[0, 0].set_ylabel('Power [dB]',fontsize = font)
axs[0, 0].set_title('Original model without random activity', fontsize = font)
axs[0, 0].tick_params(labelsize = font)
axs[0, 0].grid()

# axs[2, 0].set_xlabel("Time [s]", fontsize = font)
# axs[2, 1].set_xlabel("Time [s]", fontsize = font)



# plt.savefig("figures\\Interference_samples_Scenario_{}_{}.pdf".format(scenarios[0], scenarios[-1]),  bbox_inches='tight')
plt.show()


