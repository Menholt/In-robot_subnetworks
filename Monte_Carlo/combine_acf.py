# -*- coding: utf-8 -*-
"""
Created on Mon May 30 08:20:54 2022

@author: EG
"""

import numpy as np
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.special as special


N_samples = 100
int_samples = 100
n_iter = 16
vel = 2
fc = 3e9
c = 299792458
var = 0.7
M = 16
N_sg = 4
t_samples = 16
taus = np.linspace(0.0, 0.05, 16)
cucu_1 = np.zeros((n_iter, t_samples*2-1))
cucu_2 = np.zeros((n_iter, t_samples*2-1))
scu = np.zeros((n_iter, t_samples*2-1))
ss_1 = np.zeros((n_iter, t_samples*2-1))
ss_2 = np.zeros((n_iter, t_samples*2-1))

hf = h5py.File('data/monte_cov_parts_iter_{}_Nsamples_{}_intsamples_{}.h5'.format(n_iter, N_samples, int_samples), 'r')
for i in range(n_iter):
    cucu_1[i, :t_samples] = hf.get('cucu_1_{}'.format(i))
    cucu_2[i, :t_samples] = hf.get('cucu_2_{}'.format(i))
    scu[i, :t_samples] = hf.get('scu_{}'.format(i))
    ss_1[i, :t_samples] = hf.get('ss_1_{}'.format(i))
    ss_2[i, :t_samples] = hf.get('ss_2_{}'.format(i))
hf.close()

# hf = h5py.File('data/monte_cov_parts_iter_{}_Nsamples_{}_intsamples_{}_v2.h5'.format(n_iter, N_samples, int_samples), 'r')
# for i in range(n_iter):
#     cucu_1[i, t_samples:] = hf.get('cucu_1_{}'.format(i))[1:]
#     cucu_2[i, t_samples:] = hf.get('cucu_2_{}'.format(i))[1:]
#     scu[i, t_samples:] = hf.get('scu_{}'.format(i))[1:]
#     ss_1[i, t_samples:] = hf.get('ss_1_{}'.format(i))[1:]
#     ss_2[i, t_samples:] = hf.get('ss_2_{}'.format(i))[1:]
# hf.close()


cov_scenario_1 = np.zeros(len(taus))
N_sg = 4
for i in range(len(taus)):
    for j in range(n_iter):
        cov_scenario_1[i] += ((special.jv(0, 2*np.pi*taus[i]*vel*fc/c)**2)*var+1)*(N_sg-1)/(M-1)*ss_1[j,i]\
            +(N_sg-1)*(N_sg-2)/((M-1)*(M-2))*ss_2[j,i]
            
cov_scenario_2 = np.zeros(len(taus))
N_sg = 16
for i in range(len(taus)):
    for j in range(n_iter):
        cov_scenario_2[i] += ((special.jv(0, 2*np.pi*taus[i]*vel*fc/c)**2)*var+1)*(N_sg-1)/(M-1)*ss_1[j,i]\
            +(N_sg-1)*(N_sg-2)/((M-1)*(M-2))*ss_2[j,i]


cov_scenario_3 = np.zeros(len(taus))
N_sg = 4
for i in range(len(taus)):
    for j in range(n_iter):
        cov_scenario_3[i] += ((special.jv(0, 2*np.pi*taus[i]*vel*fc/c)**2)*var+1)*(N_sg-1)/(2*M-2)*cucu_1[j,i]\
            +(N_sg-1)*(N_sg-2)/(4*(M-1)*(M-2))*cucu_2[j,i]
        cov_scenario_3[i] += ((special.jv(0, 2*np.pi*taus[i]*vel*fc/c)**2)*var+1)*(N_sg-1)/(2*M-2)*ss_1[j,i]\
            +(N_sg-1)*(N_sg-2)/(4*(M-1)*(M-2))*ss_2[j,i]
        cov_scenario_3[i] += (N_sg-1)*(N_sg-2)/(2*(M-1)*(M-2))*scu[j,i]
        
cov_scenario_4 = np.zeros(len(taus))
N_sg = 16
for i in range(len(taus)):
    for j in range(n_iter):
        cov_scenario_4[i] += ((special.jv(0, 2*np.pi*taus[i]*vel*fc/c)**2)*var+1)*(N_sg-1)/(2*M-2)*cucu_1[j,i]\
            +(N_sg-1)*(N_sg-2)/(4*(M-1)*(M-2))*cucu_2[j,i]
        cov_scenario_4[i] += ((special.jv(0, 2*np.pi*taus[i]*vel*fc/c)**2)*var+1)*(N_sg-1)/(2*M-2)*ss_1[j,i]\
            +(N_sg-1)*(N_sg-2)/(4*(M-1)*(M-2))*ss_2[j,i]
        cov_scenario_4[i] += (N_sg-1)*(N_sg-2)/(2*(M-1)*(M-2))*scu[j,i]


mean_1 = 0.00852511016826366
mean_2 = 0.04262358748180177
mean_3 = 0.007065615498377841
mean_4 = 0.03532645025855675
mean_5 = 0.03019951720402016
# factor_1 = 1/5
# factor_2 = 1/2.5
# factor_3 = 1/11
# factor_4 = 1/5.5
# factor_5 = 1/5
factor_1 = 1
factor_2 = 1
factor_3 = 1
factor_4 = 1
factor_5 = 1
cov_scenario_1 *= factor_1
cov_scenario_2 *= factor_2
cov_scenario_3 *= factor_3
cov_scenario_5 = cov_scenario_4*factor_5
cov_scenario_4 *= factor_4


acf_1 = (cov_scenario_1-mean_1**2)/(cov_scenario_1[0]-mean_1**2)
acf_2 = (cov_scenario_2-mean_2**2)/(cov_scenario_2[0]-mean_2**2)
acf_3 = (cov_scenario_3-mean_3**2)/(cov_scenario_3[0]-mean_3**2)
acf_4 = (cov_scenario_4-mean_4**2)/(cov_scenario_4[0]-mean_4**2)
acf_5 = (cov_scenario_5-mean_4**2)/(cov_scenario_5[0]-mean_4**2)

hf = h5py.File('data/monte_acfs_iter_{}_Nsamples_{}_intsamples_{}.h5'.format(n_iter, N_samples, int_samples), 'w')
hf.create_dataset('acf_1', data = acf_1)
hf.create_dataset('acf_2', data = acf_2)
hf.create_dataset('acf_3', data = acf_3)
hf.create_dataset('acf_4', data = acf_4)
hf.create_dataset('acf_5', data = acf_5)
hf.create_dataset('taus', data = taus)
hf.close()

plt.plot(acf_1)
plt.title("scenario 1")
plt.grid()
plt.show()
plt.plot(acf_2)
plt.title("scenario 2")
plt.grid()
plt.show()
plt.plot(acf_3)
plt.title("scenario 3")
plt.grid()
plt.show()
plt.plot(acf_4)
plt.title("scenario 4")
plt.grid()
plt.show()
plt.plot(acf_5)
plt.title("scenario 5")
plt.grid()
plt.show()


