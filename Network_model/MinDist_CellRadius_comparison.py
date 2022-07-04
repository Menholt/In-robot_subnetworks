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


def auto_correlation(x, max_predict):
    auto_corr = np.zeros(max_predict)
    auto_corr[0] = np.corrcoef(x)
    for i in range(1, max_predict):
        auto_corr[i] = np.corrcoef(x[i:],x[:-i])[0,1]*(len(x)-i)/len(x)
    return auto_corr


min_dists = [0, 0.5, 1, 1.5, 2, 2.5]
cell_radiuses = [0.5, 1.5, 2.5]


interferences = np.zeros((64, int(500*10000)))

hist = np.zeros((7, 1023))
ACFs = np.zeros((6, 3, 500))

bin_edges = np.linspace(-50, 0.5, 1024)
x = (bin_edges[1:]+bin_edges[:-1])/2

min_dist = 2.5

for i,min_dist in enumerate(min_dists):    
    for j,cell_radius in enumerate(cell_radiuses):
        print("-------------------")
        # hf = h5py.File('data\\interference_realization_64_minDist_{}_time_500_scenario_{}.h5'.format(min_dist, i+1), 'r')
        # interferences = np.array(hf.get('interference'))
        # hf.close()
        hf = h5py.File('data\\acf_minDist_{}_cell_radius_{}_scenario_1.h5'.format(min_dist, cell_radius), 'r')
        ACFs[i, j] = np.array(hf.get('acf'))
        hf.close()
        
            
        # print(10*np.log10(np.mean(interferences)))
        # interferences = interferences[interferences>1e-5]
        # interferences = interferences[interferences<1]
        # print(10*np.log10(np.mean(interferences)))
        # param = stats.norm.fit(10*np.log10(interferences))
        # loc, scale = param
        # print("loc:{}".format(loc))
        # print("Scale:{}".format(scale))
        
        
        # hist[i], _ = np.histogram(10*np.log10(interferences.reshape(-1)), bins = bin_edges, density=True)
        # normal_fit = stats.norm.pdf(x, loc, scale)
        # print("KL-dist:{}".format(stats.entropy(hist[i], normal_fit)))


font = 25
# #plt.tight_layout()
# plt.figure(figsize=(12,6))
# for i in range(7):
#     plt.plot(x, hist[i])
# # plt.ylim((0.00,0.06))
# # plt.semilogx()
# # plt.xticks(ticks = np.logspace(-6, -0, 7), labels=10*np.log10(np.logspace(-6, -0, 7)))
# # plt.title('Histogram of the interference at a single control unit for the different scenarios', fontsize=font)
# plt.legend(('1', '2', '3', '4', '5', '6', '7'),fontsize=font-10)#, loc='upper right', bbox_to_anchor=(0.45, 1.18), frameon=False,fontsize=font,ncol=3, columnspacing=0.5,handletextpad=0.3 )
# plt.xlabel(r'Power [dB]',fontsize = font)
# plt.ylabel('Normalized frequency',fontsize = font)
# plt.tick_params(labelsize = font-3)
# # plt.gca().yaxis.grid(True)
# plt.savefig('figures\\Histogram_all_scenarios_minDist_{}.pdf'.format(min_dist),bbox_inches='tight')
# plt.show()


for j in range(6):
    x_acf = np.linspace(0,50,500)
    plt.figure(figsize=(12,6))
    for i in range(3):
        plt.plot(x_acf[1:], ACFs[j,i,1:])
    plt.title("Min dist of {}".format(min_dists[j]), fontsize = font)
    plt.legend(list(map(str,cell_radiuses)), title="Cell radius",fontsize=font-10, title_fontsize = font-10, loc='upper right')#, bbox_to_anchor=(0.45, 1.18), frameon=False,fontsize=font,ncol=3, columnspacing=0.5,handletextpad=0.3 )
    plt.xlabel(r'lag [ms]',fontsize = font)
    plt.ylabel('ACFs',fontsize = font)
    plt.ylim((0, 1))
    plt.tick_params(labelsize = font-3)
    # plt.savefig('figures\\acfs_scenarios_minDist_{}.pdf'.format(min_dist), bbox_inches='tight')
    plt.show()


