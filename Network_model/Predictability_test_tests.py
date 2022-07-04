# -*- coding: utf-8 -*-
"""
Created on Sat Jan  8 11:55:34 2022

@author: EG
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.stats as stats
from tqdm import tqdm
import scipy.special as special


bin_edges = np.linspace(-50, 0, 1024)
x = (bin_edges[1:]+bin_edges[:-1])/2

vel = 2
min_dist = 3
cell_radius = 2
t = 30
realizations = 1024
n_scenarios = 5
fs = 10000

hist = np.zeros((n_scenarios, 1023))
ACFs = np.zeros((n_scenarios, 1000))
log_norm_param = np.zeros((n_scenarios, 2))
normal_fits = np.zeros((n_scenarios, 1023))

for scenario in tqdm(range(1, n_scenarios + 1)):
    print("-------------------")
    
    print("Scenario nr. {}".format(scenario))
    hf = h5py.File('data\\scenario_{}\\interference_realization_{}_minDist_{}_cell_radius_{}_vel_{}_time_{}_fs_{}_scenario_{}.h5'.format(scenario, realizations, min_dist, cell_radius, vel, t, fs, scenario), 'r')
    interferences = np.array(hf.get('interference'))[:,:-1]
    hf.close()
    
    hf = h5py.File('data\\scenario_{}\\acf_minDist_{}_cell_radius_{}_vel_{}_fs_{}_scenario_{}.h5'.format(scenario, min_dist, cell_radius, vel, fs, scenario), 'r')
    ACFs[scenario-1] = np.mean(np.array(hf.get('acf')), axis = 0)
    hf.close()
    
    temp = interferences
    for i in range(len(interferences)):
        if np.mean(10*np.log10(interferences[len(interferences)-i-1,:])) <= -150:
            temp = np.delete(temp, len(interferences)-i-1, 0)
    interferences = temp

    print("Smallest interference value: {} dB".format(10*np.log10(np.min(interferences))))
    print("Mean interference without edges: {}".format(10*np.log10(np.mean(interferences))))
    param = stats.norm.fit(10*np.log10(interferences))
    log_norm_param[scenario-1] = param
    print("loc:{}".format(log_norm_param[scenario-1, 0]))
    print("Scale:{}".format(log_norm_param[scenario-1, 1]))
    
    
    hist[scenario-1], _ = np.histogram(10*np.log10(interferences.reshape(-1)), bins = bin_edges, density=True)
    normal_fits[scenario-1] = stats.norm.pdf(x, log_norm_param[scenario-1, 0], log_norm_param[scenario-1, 1])
    print("KL-dist:{}".format(stats.entropy(hist[scenario-1], normal_fits[scenario-1])))


#%%

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'k','tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'k']
linestyles = ['solid', 'solid', 'solid', 'solid', 'solid', 'dashed', 'dashed', 'dashed', 'dashed', 'dashed']


font = 25
plt.figure(figsize=(18,6))
for scenario in range(n_scenarios):
    plt.plot(x, hist[scenario], color = colors[scenario], linestyle = linestyles[scenario])
plt.legend(('1', '2', '3', '4', '5'),fontsize=font-5, title = "Scenarios", title_fontsize=font-5)#, loc='upper right', bbox_to_anchor=(0.45, 1.18), frameon=False,fontsize=font,ncol=3, columnspacing=0.5,handletextpad=0.3 )
plt.xlabel(r'Power [dB]',fontsize = font)
plt.ylabel('Normalized frequency',fontsize = font)
plt.tick_params(labelsize = font)
# plt.gca().yaxis.grid(True)
plt.tight_layout()
plt.savefig('figures\\Histogram_scenarios.pdf',bbox_inches='tight')
plt.show()


x_acf = np.linspace(0,99,1000)
plt.figure(figsize=(18,6))
for scenario in range(n_scenarios):
    plt.plot(x_acf[:], ACFs[scenario,:], color = colors[scenario], linestyle = linestyles[scenario])
plt.legend(('1', '2', '3', '4', '5'),fontsize=font-5, loc='upper right', title = "Scenarios", title_fontsize=font-5)#, bbox_to_anchor=(0.45, 1.18), frameon=False,fontsize=font,ncol=3, columnspacing=0.5,handletextpad=0.3 )
plt.xlabel(r'Lag [ms]',fontsize = font)
plt.ylabel('ACFs',fontsize = font)
plt.tick_params(labelsize = font)
plt.grid()
plt.ylim(0,1)
plt.tight_layout()
plt.savefig('figures\\acfs_scenarios.pdf', bbox_inches='tight')
plt.show()

#%%

for scenario in [1,2]:
    hf = h5py.File('data\\interference_realization_{}_minDist_{}_cell_radius_{}_vel_{}_time_{}_scenario_{}.h5'.format(realizations, min_dist, cell_radius, vel, t,scenario), 'r')
    interferences = np.array(hf.get('interference'))[:,:-1]
    hf.close()
    
    hf = h5py.File('data\\acf_minDist_{}_cell_radius_{}_vel_{}_scenario_{}.h5'.format(min_dist, cell_radius, vel, scenario), 'r')
    acf = np.mean(np.array(hf.get('acf')), axis = 0)
    hf.close()
    
    hist, _ = np.histogram(10*np.log10(interferences.reshape(-1)), bins = bin_edges, density=True)
    
    plt.figure(figsize=(9,6))
    plt.plot(x, hist, linestyle = linestyles[scenario])
    plt.xlabel(r'Power [dB]',fontsize = font)
    plt.ylabel('Normalized frequency',fontsize = font)
    plt.tick_params(labelsize = font)
    # plt.gca().yaxis.grid(True)
    plt.tight_layout()
    plt.savefig('figures\\Histogram_initial_model_{}.pdf'.format(scenario),bbox_inches='tight')
    plt.show()
    
    
    x_acf = np.linspace(0,49,500)
    plt.figure(figsize=(9,6))
    plt.plot(x_acf[:], acf, linestyle = linestyles[scenario])
    plt.xlabel(r'Lag [ms]',fontsize = font)
    plt.ylabel('ACFs',fontsize = font)
    plt.ylim(0,1)
    plt.tick_params(labelsize = font)
    plt.grid()
    plt.tight_layout()
    plt.savefig('figures\\acf_initial_model_{}.pdf'.format(scenario), bbox_inches='tight')
#     plt.show()