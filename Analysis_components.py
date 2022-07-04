# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 10:52:29 2022

@author: lukas
"""

import numpy as np
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
#%%
realizations = 64
hf = h5py.File('data/interference_plus_resten_realization_64_minDist_2.5_cell_radius_2.5_time_500_scenario_1_newer.h5'.format(realizations), 'r')
stats = np.array(hf.get('interference'))
hf.close()

fs = 10000
t = 500

interference = np.zeros((realizations, t*fs))
path_l = np.zeros((realizations, t*fs))
ss_fading = np.zeros((realizations, t*fs))
shadow = np.zeros((realizations, t*fs))

# for i in tqdm(range(realizations)):
#     interference[i] = stats[i,:]
for i in tqdm(range(realizations)):
    interference[i] = stats[i,:,0]
    path_l[i] = stats[i,:,1]
    ss_fading[i] = stats[i,:,2]
    shadow[i] = stats[i,:,3]
#%%
int_len = 2000
start_time = 0
snapshot = 1
t = np.linspace(start_time/fs, (start_time + int_len)/fs, int_len)

font = 25
# plt.figure(figsize=(12,4))
# plt.plot(t, 10*np.log10(interference[snapshot, start_time:start_time+int_len]))
# plt.xlabel("Time [s]", fontsize=font)
# plt.ylabel("Interference [dB]", fontsize=font)
# plt.xticks(fontsize=font-10)
# plt.yticks(fontsize=font-10)
# plt.tight_layout()
# # plt.savefig("figures/interference_fra_en_enkelt_start_{}_len_{}.pdf".format(start_time, int_len))
# plt.show()

interference = path_l*np.sqrt(ss_fading)*shadow
plt.figure(figsize=(12,4))
plt.plot(t, 10*np.log10(interference[snapshot, start_time:start_time+int_len]))
plt.xlabel("Time [s]", fontsize=font)
plt.ylabel("Interference [dB]", fontsize=font)
plt.xticks(fontsize=font-10)
plt.yticks(fontsize=font-10)
plt.tight_layout()
plt.savefig("figures/interference_fra_en_enkelt_start_{}_len_{}.pdf".format(start_time, int_len))
plt.show()


plt.figure(figsize=(12,4))
plt.plot(t, 10*np.log10(path_l[snapshot, start_time:start_time+int_len]))
plt.xlabel("Time [s]", fontsize=font)
plt.ylabel("Path-loss [dB]", fontsize=font)
plt.xticks(fontsize=font-10)
plt.yticks(fontsize=font-10)
plt.tight_layout()
plt.savefig("figures/path_loss_fra_en_enkelt_start_{}_len_{}.pdf".format(start_time, int_len))
plt.show()


plt.figure(figsize=(12,4))
plt.plot(t, 10*np.log10(np.sqrt(ss_fading[snapshot, start_time:start_time+int_len])))
plt.xlabel("Time [s]", fontsize=font)
plt.ylabel("Ss fading [dB]", fontsize=font)
plt.xticks(fontsize=font-10)
plt.yticks(fontsize=font-10)
plt.tight_layout()
plt.savefig("figures/ss_fading_fra_en_enkelt_start_{}_len_{}.pdf".format(start_time, int_len))
plt.show()


plt.figure(figsize=(12,4))
plt.plot(t, 10*np.log10(shadow[snapshot, start_time:start_time+int_len]))
plt.xlabel("Time [s]", fontsize=font)
plt.ylabel("Shadowing [dB]", fontsize=font)
plt.xticks(fontsize=font-10)
plt.yticks(fontsize=font-10)
plt.tight_layout()
plt.savefig("figures/shadow_fra_en_enkelt_start_{}_len_{}.pdf".format(start_time, int_len))
plt.show()

#%%

plt.hist(10*np.log10(interference[interference>0].reshape(-1)), bins = 1024)
# plt.savefig("figures/hist_interference_fra_en_enkelt_start_{}_len_{}.pdf".format(start_time, int_len))
plt.show()

plt.hist(10*np.log10(path_l[snapshot]), bins = 1024)
# plt.savefig("figures/hist_path_loss_fra_en_enkelt_start_{}_len_{}.pdf".format(start_time, int_len))
plt.show()

plt.hist(ss_fading[:].reshape(-1), bins = 1024)
# plt.savefig("figures/hist_ss_fading_fra_en_enkelt_start_{}_len_{}.pdf".format(start_time, int_len))
plt.show()

plt.hist(10*np.log10(shadow[snapshot]), bins = 1024)
# plt.savefig("figures/hist_shadow_fra_en_enkelt_start_{}_len_{}.pdf".format(start_time, int_len))
plt.show()