# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 11:10:21 2022

@author: EG
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def createMap(width, height, sigmaS, correlationDistance, stepsize):
    num_x_points = int(width/stepsize) + 1
    num_y_points = int(height/stepsize) + 1
    mapXPoints=np.linspace(0, width, num=num_x_points, endpoint=True)
    mapYPoints=np.linspace(0, height, num=num_y_points, endpoint=True)
    
    N1 = len(mapXPoints)
    N2 = len(mapYPoints)
    G = np.zeros([N1,N2],dtype=np.float64)
    for n in range(N1):
        for m in range(N2):
            G[n,m]= sigmaS*np.exp(-1*np.sqrt(np.min([np.absolute(mapXPoints[0]-mapXPoints[n]),\
                                      width-np.absolute(mapXPoints[0]-mapXPoints[n])])**2\
            + np.min([np.absolute(mapYPoints[0]-mapYPoints[m]),height\
                  -np.absolute(mapYPoints[0]-mapYPoints[m])])**2)/correlationDistance)
    Gamma = np.fft.fft2(G)
    Z = np.random.randn(N1,N2) + 1j*np.random.randn(N1,N2)
    mapp = np.real(np.fft.fft2(np.multiply(np.sqrt(Gamma),Z)\
                               /np.sqrt(N1*N2)))
    
    return mapp



def shadowing(grf, delta, loc1, loc2, stepsize, height, width):
    grf1 = grf[int((loc1[0]+width/2)/stepsize)][int((loc1[1]+height/2)/stepsize)]
    grf2 = grf[int((loc2[0]+width/2)/stepsize)][int((loc2[1]+height/2)/stepsize)]
    dist = np.linalg.norm(loc1-loc2)
    return (((1-np.exp(-dist/delta))/(np.sqrt(2)*np.sqrt(1+np.exp(-dist/delta)))*(grf1+grf2))), dist


# seed = np.random.randint(0,1e6)
seed = 79709
np.random.seed(seed)

height = 30
width = 30
stepsize = 1/20
delta = 5
sigma = 3

grfs = np.zeros((1024, 361201))

means = []
varis = []
for i in tqdm(range(1024)):
    grf = createMap(width, height, sigma, delta, stepsize)
    grfs[i] = grf.reshape(-1)
    means.append(np.mean(grfs[i]))
    varis.append(np.var(grfs[i]))
#%%
N=25000
shadow_scatter = np.zeros((N,2))
loc1 = np.random.uniform(-15,15, (N,2))
loc2 = np.random.uniform(-15,15, (N,2))
for i in range(N):
    shadow_scatter[i] = shadowing(grf, delta, loc1[i], loc2[i], stepsize, height, width)

font = 25

fig, axs = plt.subplots(1, 2, sharey=True)
fig.set_figheight(6)
fig.set_figwidth(18)

bins_mean = np.linspace(np.min(means), np.max(means), 16)
bins_var = np.linspace(np.min(varis), np.max(varis), 16)

axs[0].hist(means, bins = bins_mean, density=True)
axs[0].set_xlabel("Means", fontsize = font)
axs[0].set_ylabel("Normalized frequency", fontsize = font)
axs[0].tick_params(labelsize = font)

axs[1].hist(varis, bins = bins_var, density=True)
axs[1].set_xlabel("Variances", fontsize = font)
axs[1].tick_params(labelsize = font)

plt.tight_layout=True
plt.savefig("figures/shadow_means_var_hists.pdf", bbox_inches = "tight")
plt.show()



plt.figure(figsize=(18,6))
plt.scatter(shadow_scatter[:,1], shadow_scatter[:,0])
# plt.ylim(0,4)
plt.xlabel("Distance [m]", fontsize = font)
plt.ylabel("Shadow value [dB]", fontsize = font)
plt.xticks(fontsize=font)
plt.yticks(fontsize=font)
plt.tight_layout=True
plt.savefig('figures\\Shadowing_scatter.pdf', bbox_inches='tight')
plt.show()


x=np.linspace(-15, 15, int(width/stepsize +1))
y=np.linspace(-15, 15, int(height/stepsize + 1))
plt.figure(figsize=(12,6))
plt.pcolormesh(x,y,grfs[0].reshape(601, 601))
plt.xlabel("width [m]", fontsize = font)
plt.ylabel("height [m]", fontsize = font)
plt.xticks(fontsize=font)
plt.yticks(fontsize=font)
cb = plt.colorbar()
cb.ax.tick_params(labelsize = font)
plt.tight_layout=True
plt.savefig('figures\\ShadowFading_map_sigma_{}_delta_{}.pdf'.format(sigma, delta),bbox_inches='tight')
plt.show()


plt.figure(figsize=(12,6))
plt.hist(grfs[0], bins=256, density = True)
x_axis = "Shadow map values"
y_axis = "Normalized frequency"
plt.xlabel(x_axis, fontsize = font)
plt.ylabel(y_axis, fontsize = font)
plt.xticks(fontsize=font)
plt.yticks(fontsize=font)
plt.tight_layout=True
plt.savefig('figures\\histogram_of_shadow_map.pdf',bbox_inches='tight')
plt.show()
# print(np.mean(grfs))
# print(np.var(grfs))
# print(seed)
