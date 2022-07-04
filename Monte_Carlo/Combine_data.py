# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 09:05:20 2022

@author: EG
"""
import numpy as np
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
# monte_mean_mindist25 = np.zeros(16)
# monte_mean_mindist2 = np.zeros(16)
# monte_mean_mindist15 = np.zeros(16)
# monte_mean_mindist1 = np.zeros(16)


# for i in range(16):
#     hf = h5py.File('data\\monte_mean_realization_1_minDists_(2.5, 2, 1.5, 1)_v{}.h5'.format(i), 'r')
#     monte_mean_mindist25[i] = np.array(hf.get('monte_mean'))[0]
#     monte_mean_mindist2[i] = np.array(hf.get('monte_mean'))[1]
#     monte_mean_mindist15[i] = np.array(hf.get('monte_mean'))[2]
#     monte_mean_mindist1[i] = np.array(hf.get('monte_mean'))[3]
#     hf.close()

# hf = h5py.File('data\\monte_mean_realization_16_minDist_2.5.h5', 'w')
# hf.create_dataset('monte_mean', data = monte_mean_mindist25)
# hf.close()
# hf = h5py.File('data\\monte_mean_realization_16_minDist_2.h5', 'w')
# hf.create_dataset('monte_mean', data = monte_mean_mindist2)
# hf.close()
# hf = h5py.File('data\\monte_mean_realization_16_minDist_1.5.h5', 'w')
# hf.create_dataset('monte_mean', data = monte_mean_mindist15)
# hf.close()
# hf = h5py.File('data\\monte_mean_realization_16_minDist_1.h5', 'w')
# hf.create_dataset('monte_mean', data = monte_mean_mindist1)
# hf.close()

# hf = h5py.File('data\\monte_mean_realization_16_minDists_(0.5, 0).h5', 'r')
# monte_mean_mindist05 = np.array(hf.get('monte_mean'))[:16]
# monte_mean_mindist0 = np.array(hf.get('monte_mean'))[16:]
# hf.close()

# hf = h5py.File('data\\monte_mean_realization_16_minDist_0.5.h5', 'w')
# hf.create_dataset('monte_mean', data = monte_mean_mindist05)
# hf.close()
# hf = h5py.File('data\\monte_mean_realization_16_minDist_0.h5', 'w')
# hf.create_dataset('monte_mean', data = monte_mean_mindist0)
# hf.close()

#%% combine acf


min_dist = 3
cell_radius = 2

t = 30
scenario = 6

N_samples = 100
int_samples = 100
n_iter = 16

t_samples = 16

cov_MC = np.zeros((n_iter, t_samples))
# cov_cucu = np.zeros((n_iter, 2*t_samples-1))
# # cov_cus = np.zeros((n_iter, 2*t_samples))
# cov_scu = np.zeros((n_iter, 2*t_samples-1))
# cov_ss = np.zeros((n_iter, 2*t_samples-1))

hf = h5py.File('data/monte_cov_iter_{}_Nsamples_{}_intsamples_{}_4_5.h5'.format(n_iter, N_samples, int_samples), 'r')
# hf_cucu = h5py.File('data/monte_cov_cucu_iter_{}_Nsamples_{}_intsamples_{}.h5'.format(n_iter, N_samples, int_samples), 'r')
# # hf_cus = h5py.File('data/monte_cov_cus_iter_{}_Nsamples_{}_intsamples_{}.h5'.format(n_iter, N_samples, int_samples), 'r')
# hf_scu = h5py.File('data/monte_cov_scu_iter_{}_Nsamples_{}_intsamples_{}.h5'.format(n_iter, N_samples, int_samples), 'r')
# hf_ss = h5py.File('data/monte_cov_ss_iter_{}_Nsamples_{}_intsamples_{}.h5'.format(n_iter, N_samples, int_samples), 'r')
for i in range(n_iter):
    cov_MC[i,:t_samples] = np.array(hf.get('covariance_{}'.format(i)))
    # cov_cucu[i,:t_samples] = np.array(hf_cucu.get('covariance_{}'.format(i)))
    # # cov_cus[i] = np.array(hf_cus.get('covariance_{}'.format(i)))
    # cov_scu[i,:t_samples] = np.array(hf_scu.get('covariance_{}'.format(i)))
    # cov_ss[i,:t_samples] = np.array(hf_ss.get('covariance_{}'.format(i)))
hf.close()
# hf_cucu.close()
# # hf_cus.close()
# hf_scu.close()
# hf_ss.close()



# hf = h5py.File('data/monte_cov_iter_{}_Nsamples_{}_intsamples_{}_v2.h5'.format(n_iter, N_samples, int_samples), 'r')
# hf_cucu = h5py.File('data/monte_cov_cucu_iter_{}_Nsamples_{}_intsamples_{}_v2.h5'.format(n_iter, N_samples, int_samples), 'r')
# # hf_cus = h5py.File('data/monte_cov_cus_iter_{}_Nsamples_{}_intsamples_{}.h5'.format(n_iter, N_samples, int_samples), 'r')
# hf_scu = h5py.File('data/monte_cov_scu_iter_{}_Nsamples_{}_intsamples_{}_v2.h5'.format(n_iter, N_samples, int_samples), 'r')
# hf_ss = h5py.File('data/monte_cov_ss_iter_{}_Nsamples_{}_intsamples_{}_v2.h5'.format(n_iter, N_samples, int_samples), 'r')
# for i in range(n_iter):
#     cov_MC[i,t_samples:] = np.array(hf.get('covariance_{}'.format(i)))[2:]
#     cov_cucu[i,t_samples:] = np.array(hf_cucu.get('covariance_{}'.format(i)))[2:]
#     # cov_cus[i] = np.array(hf_cus.get('covariance_{}'.format(i)))
#     cov_scu[i,t_samples:] = np.array(hf_scu.get('covariance_{}'.format(i)))[2:]
#     cov_ss[i,t_samples:] = np.array(hf_ss.get('covariance_{}'.format(i)))[2:]
# hf.close()
# hf_cucu.close()
# # hf_cus.close()
# hf_scu.close()
# hf_ss.close()

faktor = 1/11

cov_MC = np.mean(cov_MC, axis=0)*faktor
# cov_cucu = np.mean(cov_cucu, axis=0)
# # cov_cus = np.mean(cov_cus, axis=0)
# cov_scu = np.mean(cov_scu, axis=0)
# cov_ss = np.mean(cov_ss, axis=0)

mean = 0.007065615498377841
mean = 0.03532645025855675

acf_MC = (cov_MC-mean**2)/(cov_MC[0]-mean**2)
# acf_cucu = (cov_cucu-mean**2)/(cov_cucu[0]-mean**2)
# acf_scu = (cov_scu-mean**2)/(cov_scu[0]-mean**2)
# acf_ss = (cov_ss-mean**2)/(cov_ss[0]-mean**2)

taus = np.linspace(0, 0.05, 16)

# hf = h5py.File('data/monte_acf_iter_{}_Nsamples_{}_intsamples_{}_4_5.h5'.format(n_iter, N_samples, int_samples), 'w')
# hf.create_dataset('acf_MC', data = acf_MC)
# # hf.create_dataset('acf_cucu', data = acf_cucu)
# # hf.create_dataset('acf_scu', data = acf_scu)
# # hf.create_dataset('acf_ss', data = acf_ss)
# hf.create_dataset('taus', data = taus)
# hf.close()
plt.plot(acf_MC)
#%%

min_dist = 3
cell_radius = 2

t = 30
scenario = 6

N_samples = 100
int_samples = 100
n_iter = 16

t_samples = 16

# cov_MC = np.zeros((n_iter, 2*t_samples-1))
cov_MC = np.zeros((n_iter, t_samples))


hf = h5py.File('data/monte_cov_sensor_iter_{}_Nsamples_{}_intsamples_{}_start_scenario_1.h5'.format(n_iter, N_samples, int_samples), 'r')
for i in range(n_iter):
    cov_MC[i,:t_samples] = np.array(hf.get('covariance_{}'.format(i)))

hf.close()



# hf = h5py.File('data/monte_cov_sensor_iter_{}_Nsamples_{}_intsamples_{}.h5'.format(n_iter, N_samples, int_samples), 'r')
# for i in range(n_iter):
#     cov_MC[i,t_samples:] = np.array(hf.get('covariance_{}'.format(i)))[2:]
# hf.close()
mean = 0.00852511016826366
print(mean**2)
# faktor = 5.5

cov_MC = np.mean(cov_MC, axis=0)*faktor

acf_MC = (cov_MC-mean**2)/(cov_MC[0]-mean**2)

taus = np.linspace(0, 0.1, 31)

# hf = h5py.File('data/monte_acf_sensor_iter_{}_Nsamples_{}_intsamples_{}_scenario_1.h5'.format(n_iter, N_samples, int_samples), 'w')
# hf.create_dataset('acf_MC', data = acf_MC)
# # hf.create_dataset('acf_cucu', data = acf_cucu)
# # hf.create_dataset('acf_scu', data = acf_scu)
# # hf.create_dataset('acf_ss', data = acf_ss)
# hf.create_dataset('taus', data = taus)
# hf.close()
plt.plot(acf_MC)
plt.grid()

#%% combine acf


min_dist = 3
cell_radius = 2

t = 30
scenario = 6

N_samples = 100
int_samples = 100
n_iter = 16

t_samples = 16

cov_MC = np.zeros((n_iter, t_samples*2-1))
cov_cucu = np.zeros((n_iter, 2*t_samples-1))
# cov_cus = np.zeros((n_iter, 2*t_samples))
cov_scu = np.zeros((n_iter, 2*t_samples-1))
cov_ss = np.zeros((n_iter, 2*t_samples-1))

hf = h5py.File('data/monte_cov_iter_{}_Nsamples_{}_intsamples_{}.h5'.format(n_iter, N_samples, int_samples), 'r')
hf_cucu = h5py.File('data/monte_cov_cucu_iter_{}_Nsamples_{}_intsamples_{}.h5'.format(n_iter, N_samples, int_samples), 'r')
# hf_cus = h5py.File('data/monte_cov_cus_iter_{}_Nsamples_{}_intsamples_{}.h5'.format(n_iter, N_samples, int_samples), 'r')
hf_scu = h5py.File('data/monte_cov_scu_iter_{}_Nsamples_{}_intsamples_{}.h5'.format(n_iter, N_samples, int_samples), 'r')
hf_ss = h5py.File('data/monte_cov_ss_iter_{}_Nsamples_{}_intsamples_{}.h5'.format(n_iter, N_samples, int_samples), 'r')
for i in range(n_iter):
    cov_MC[i,:t_samples] = np.array(hf.get('covariance_{}'.format(i)))
    cov_cucu[i,:t_samples] = np.array(hf_cucu.get('covariance_{}'.format(i)))
    # cov_cus[i] = np.array(hf_cus.get('covariance_{}'.format(i)))
    cov_scu[i,:t_samples] = np.array(hf_scu.get('covariance_{}'.format(i)))
    cov_ss[i,:t_samples] = np.array(hf_ss.get('covariance_{}'.format(i)))
hf.close()
hf_cucu.close()
# hf_cus.close()
hf_scu.close()
hf_ss.close()



hf = h5py.File('data/monte_cov_iter_{}_Nsamples_{}_intsamples_{}_v2.h5'.format(n_iter, N_samples, int_samples), 'r')
hf_cucu = h5py.File('data/monte_cov_cucu_iter_{}_Nsamples_{}_intsamples_{}_v2.h5'.format(n_iter, N_samples, int_samples), 'r')
# hf_cus = h5py.File('data/monte_cov_cus_iter_{}_Nsamples_{}_intsamples_{}.h5'.format(n_iter, N_samples, int_samples), 'r')
hf_scu = h5py.File('data/monte_cov_scu_iter_{}_Nsamples_{}_intsamples_{}_v2.h5'.format(n_iter, N_samples, int_samples), 'r')
hf_ss = h5py.File('data/monte_cov_ss_iter_{}_Nsamples_{}_intsamples_{}_v2.h5'.format(n_iter, N_samples, int_samples), 'r')
for i in range(n_iter):
    cov_MC[i,t_samples:] = np.array(hf.get('covariance_{}'.format(i)))[2:]
    cov_cucu[i,t_samples:] = np.array(hf_cucu.get('covariance_{}'.format(i)))[2:]
    # cov_cus[i] = np.array(hf_cus.get('covariance_{}'.format(i)))
    cov_scu[i,t_samples:] = np.array(hf_scu.get('covariance_{}'.format(i)))[2:]
    cov_ss[i,t_samples:] = np.array(hf_ss.get('covariance_{}'.format(i)))[2:]
hf.close()
hf_cucu.close()
# hf_cus.close()
hf_scu.close()
hf_ss.close()

faktor = 1/33

cov_MC = np.mean(cov_MC, axis=0)*faktor
cov_cucu = np.mean(cov_cucu, axis=0)
# cov_cus = np.mean(cov_cus, axis=0)
cov_scu = np.mean(cov_scu, axis=0)
cov_ss = np.mean(cov_ss, axis=0)

mean = 0.007065615498377841
# mean = 0.03532645025855675
print(mean**2)

acf_MC = (cov_MC-mean**2)/(cov_MC[0]-mean**2)
acf_cucu = (cov_cucu-mean**2)/(cov_cucu[0]-mean**2)
acf_scu = (cov_scu-mean**2)/(cov_scu[0]-mean**2)
acf_ss = (cov_ss-mean**2)/(cov_ss[0]-mean**2)

taus = np.linspace(0, 0.05, 16)

# hf = h5py.File('data/monte_acf_iter_{}_Nsamples_{}_intsamples_{}.h5'.format(n_iter, N_samples, int_samples), 'w')
# hf.create_dataset('acf_MC', data = acf_MC)
# # hf.create_dataset('acf_cucu', data = acf_cucu)
# # hf.create_dataset('acf_scu', data = acf_scu)
# # hf.create_dataset('acf_ss', data = acf_ss)
# hf.create_dataset('taus', data = taus)
# hf.close()
plt.plot(acf_MC)
plt.grid()


#%%
min_dist = 3
cell_radius = 2

t = 30
scenario = 6

N_samples = 100
int_samples = 100
n_iter = 16

t_samples = 16

# cov_MC = np.zeros((n_iter, 2*t_samples-1))
cov_MC = np.zeros((n_iter, t_samples))


hf = h5py.File('data/monte_cov_sensor_iter_{}_Nsamples_{}_intsamples_{}_start.h5'.format(n_iter, N_samples, int_samples), 'r')
for i in range(n_iter):
    cov_MC[i,:t_samples] = np.array(hf.get('covariance_{}'.format(i)))

hf.close()



# hf = h5py.File('data/monte_cov_sensor_iter_{}_Nsamples_{}_intsamples_{}.h5'.format(n_iter, N_samples, int_samples), 'r')
# for i in range(n_iter):
#     cov_MC[i,t_samples:] = np.array(hf.get('covariance_{}'.format(i)))[2:]
# hf.close()
mean = 0.04262358748180177

faktor = 5.5*4

cov_MC = np.mean(cov_MC, axis=0)*faktor

acf_MC = (cov_MC-mean**2)/(cov_MC[0]-mean**2)

taus = np.linspace(0, 0.1, 31)

# hf = h5py.File('data/monte_acf_sensor_iter_{}_Nsamples_{}_intsamples_{}_scenario_2.h5'.format(n_iter, N_samples, int_samples), 'w')
# hf.create_dataset('acf_MC', data = acf_MC)
# # hf.create_dataset('acf_cucu', data = acf_cucu)
# # hf.create_dataset('acf_scu', data = acf_scu)
# # hf.create_dataset('acf_ss', data = acf_ss)
# hf.create_dataset('taus', data = taus)
# hf.close()
plt.plot(acf_MC)
plt.grid()
