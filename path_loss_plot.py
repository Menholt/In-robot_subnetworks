# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 10:22:53 2022

@author: EG
"""
import numpy as np 
import matplotlib.pyplot as plt

def path_loss(d, alpha):
    return min(1, np.linalg.norm(d)**(-alpha))

dists = np.linspace(0.01,10,1000)
alpha = 3

font = 25

fig = plt.figure()
fig.set_figheight(6)
fig.set_figwidth(18)
plt.tight_layout=True
plt.tick_params(labelsize = font)
plt.grid(True)
pl = np.zeros_like(dists)
for i in range(len(dists)):
    pl[i] = path_loss(dists[i], alpha)

plt.plot(dists, 10*np.log10(pl))

plt.ylabel("Path loss [dB]", fontsize = font)  
plt.xlabel("Distance [m]", fontsize = font)  
plt.savefig("figures\\path_loss.pdf", bbox_inches='tight')        
plt.show()