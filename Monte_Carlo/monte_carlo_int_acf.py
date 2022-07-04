# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 11:16:19 2022

@author: EG
"""

import numpy as np
from tqdm import tqdm
import scipy.special as special
from Scripts.Network_model import interference_functions as i_f
import h5py
import multiprocessing as mp

def monte_int(height, width, n_nodes, N_samples, int_samples, N_sg, min_dist, r, taus, fs, vel, c, fc, sigma, delta, stepsize):
    summ = np.zeros(len(taus))
    cucu = np.zeros(len(taus))
    # cus = np.zeros(len(taus))
    scu = np.zeros(len(taus))
    ss = np.zeros(len(taus))
     
    # means = np.zeros(len(taus))
    for i in tqdm(range(N_samples)):
        grf = i_f.createMap(width, height, sigma, delta, stepsize)
        x_t1 = np.zeros((n_nodes,2))
        for j in range(n_nodes):
            x_t1[j, 0] = np.random.uniform(-width/2 + r, width/2 - r)
            x_t1[j, 1] = np.random.uniform(-height/2 + r, height/2 - r)
            k = 0
            while k < j:
                if np.linalg.norm(x_t1[j]-x_t1[k])<min_dist:
                    x_t1[j, 0] = np.random.uniform(-width/2 + r, width/2 - r)
                    x_t1[j, 1] = np.random.uniform(-height/2 + r, height/2 - r)
                    k = 0
                else:
                    k += 1
        x_t2 = np.copy(x_t1)
        
        directions_t1 = np.random.uniform(0, 2*np.pi, (n_nodes))        
        
        displacement = np.zeros((int_samples,len(x_t1),2))+r
        displacement_t1 = np.zeros((int_samples,len(x_t1),2))+r
        displacement_t2 = np.zeros((int_samples,len(x_t1),2))+r
        for j in range(int_samples):
            for k in range(len(x_t1)):
                while np.linalg.norm(displacement[j,k])>=r:
                    displacement[j,k] = np.random.uniform(-r, r, 2)
        
                # displacement_t1[j, k] = rotate(displacement[j, k], directions_t1[k])

        directions_temp = directions_t1
        for j in range(len(taus)):
            for __ in range(int(fs*taus[1])):
                x_t2, directions_temp = i_f.random_direction(x_t2, directions_temp, vel, min_dist, r, int(1/fs), height, width)
            # directions_t2 = directions_temp
            # # for k in range(int_samples):
            # #     for n in range(len(x_t1)):
            # #         displacement_t2[k, n] = rotate(displacement[k, n], directions_t2[n])
            displacement_t1 = displacement
            displacement_t2 = displacement
            
            cucu[j] += monte_int_cucu(x_t1, x_t2, taus[j], grf, delta, stepsize, r, height, width, min_dist, vel, c, fc, N_sg, n_nodes)
            # cus[j] += monte_int_cus(x_t1, x_t2, taus[j], displacement_t2, grf, delta, stepsize, int_samples, r, height, width, min_dist, vel, N_sg, n_nodes)
            scu[j] += monte_int_scu(x_t1, x_t2, taus[j], displacement_t1, grf, delta, stepsize, int_samples, r, height, width, min_dist, vel, N_sg, n_nodes)
            ss[j] += monte_int_ss(x_t1, x_t2, taus[j], displacement_t1, displacement_t2, grf, delta, stepsize, int_samples, r, height, width, min_dist, vel, c, fc, N_sg, n_nodes)
            # summ[j] += cucu[j] + cus[j] + scu[j] + ss[j]
            summ[j] += cucu[j] + 2*scu[j] + ss[j]
    # return summ/N_samples, cucu/N_samples, cus/N_samples, scu/N_samples, ss/N_samples
    return summ/N_samples, cucu/N_samples, scu/N_samples, ss/N_samples

def monte_int_ss(x_t1, x_t2, tau, displacement_t1, displacement_t2, grf, delta, stepsize, int_samples, r, height, width, min_dist, vel, c, fc, N_sg, n_nodes):
    cov_ss = 0
    eta_ = eta(x_t1, x_t2, displacement_t1, displacement_t2, grf, delta, stepsize, int_samples, r, height, width, min_dist)
    cov_ss += ((special.jv(0, 2*np.pi*tau*vel*fc/c)**2*0.7+1)*(N_sg-1))/(2*n_nodes-2)* eta_
    cov_ss += (N_sg-1)*(N_sg-2)/(4*(n_nodes-1)*(n_nodes-2)) * kappa_alpha(x_t1, x_t2, displacement_t1, displacement_t1, r, int_samples, grf, delta, stepsize, height, width)
    return cov_ss

def monte_int_cucu(x_t1, x_t2, tau, grf, delta, stepsize, r, height, width, min_dist, vel, c, fc, N_sg, n_nodes):
    cov_cucu = 0
    for i in range(len(x_t1)-1):
        cov_cucu += (N_sg-1)/(2*n_nodes-2)*(special.jv(0, 2*np.pi*tau*vel*fc/c)**2*0.7+1)\
            *pathloss(x_t1[0] - x_t1[i+1])*i_f.shadowing(grf, delta, x_t1[i+1], x_t1[0], stepsize, height, width)\
            *pathloss(x_t2[0] - x_t2[i+1])*i_f.shadowing(grf, delta, x_t2[i+1], x_t2[0], stepsize, height, width)
        
        for j in range(len(x_t1)-1):
            if j!=i:
                cov_cucu += ((N_sg-1)*(N_sg-2))/(4*(n_nodes-1)*(n_nodes-2))\
                    *pathloss(x_t1[0] - x_t1[i+1])*i_f.shadowing(grf, delta, x_t1[i+1], x_t1[0], stepsize, height, width)\
                    *pathloss(x_t2[0] - x_t2[j+1])*i_f.shadowing(grf, delta, x_t2[j+1], x_t2[0], stepsize, height, width)
    return cov_cucu

def monte_int_cus(x_t1, x_t2, tau, displacement_t2, grf, delta, stepsize, int_samples, r, height, width, min_dist, vel, N_sg, n_nodes):
    cov_cus = 0
    for i in range(len(x_t1)-1):
        for j in range(len(x_t1)-1):
            if j!=i:
                cov_cus += pathloss(x_t1[0] - x_t1[i+1])*i_f.shadowing(grf, delta, x_t1[i+1], x_t1[0], stepsize, height, width)\
                    * kappa(x_t2, displacement_t2, int_samples, r, i, grf, delta, stepsize, height, width)
    return cov_cus * ((N_sg-1)*(N_sg-2))/(4*(n_nodes-1)*(n_nodes-2))

def monte_int_scu(x_t1, x_t2, tau, displacement_t1, grf, delta, stepsize, int_samples, r, height, width, min_dist, vel, N_sg, n_nodes):
    cov_scu = 0
    for i in range(len(x_t1)-1):
        for j in range(len(x_t1)-1):
            if j!=i:
                cov_scu += pathloss(x_t2[0] - x_t2[i+1])*i_f.shadowing(grf, delta, x_t2[i+1], x_t2[0], stepsize, height, width)\
                    * kappa(x_t1, displacement_t1, int_samples, r, i, grf, delta, stepsize, height, width)
    return cov_scu * ((N_sg-1)*(N_sg-2))/(4*(n_nodes-1)*(n_nodes-2))

def pathloss(x):
    return min(1,np.linalg.norm(x)**(-3))


def rotate(x, angle):
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return np.matmul(R, x)


def kappa_alpha(x_t1, x_t2, displacement_t1, displacement_t2, r, int_samples, grf, delta, stepsize, height, width):
    
    kappa_alpha_sum_var = 0
    for i in range(int_samples):
        for j in range(len(x_t1)-1):
            alph = kappa(x_t2, displacement_t2, int_samples, r, j, grf, delta, stepsize, height, width)
            kappa_alpha_sum_var += pathloss(x_t1[j+1] + displacement_t1[i, j+1] - x_t1[0])* i_f.shadowing(grf, delta, x_t1[j+1] + displacement_t1[i, j+1], x_t1[0], stepsize, height, width)* alph
    
    return kappa_alpha_sum_var/(r**2*np.pi*int_samples)


def kappa(x, displacement, int_samples, r, i, grf, delta, stepsize, height, width):
    for j in range(int_samples):
        kappa_sum_var = 0
        for k in range(len(x)-1):
            if k!=i:
                kappa_sum_var += pathloss(x[k+1] + displacement[j, k+1] - x[0])* i_f.shadowing(grf, delta, x[k+1] + displacement[j, k+1], x[0], stepsize, height, width)
    return kappa_sum_var/(r**2*np.pi*int_samples)


def eta(x_t1, x_t2, displacement_t1, displacement_t2, grf, delta, stepsize, int_samples, r, height, width, min_dist):
    eta_sum_var = 0
    for i in range(int_samples):
        for j in range(len(x_t1)-1):
            eta_sum_var += pathloss(x_t1[j+1] + displacement_t1[i, j+1] - x_t1[0])*i_f.shadowing(grf, delta, x_t1[j+1] + displacement_t1[i, j+1],  x_t1[0], stepsize, height, width)\
            *pathloss(x_t2[j+1] + displacement_t2[i, j+1] - x_t2[0])*i_f.shadowing(grf, delta, x_t2[j+1] + displacement_t2[i, j+1],  x_t2[0], stepsize, height, width)
    return eta_sum_var/(r**2*np.pi*int_samples)


def calc_monte_acf(in_put):
    np.random.seed()
    taus, N_samples, int_samples = in_put
    height = 30
    width = 30
    n_nodes = 16

    min_dist = 3
    r = 2 # radius of the subnetworks

    fs = 10000
    vel = 2
    N_sg = 4
    c = 299792458
    fc = 3e9
    delta = 5 # decorrelation distance
    sigma = 3 # variance
    stepsize = 1/10

    return monte_int(height, width, n_nodes, N_samples, int_samples, N_sg, min_dist, r, taus, fs, vel, c, fc, sigma, delta, stepsize)

if __name__ == "__main__":
    N = mp.cpu_count()
    
    N_samples = 100
    int_samples = 100
    
    taus = np.linspace(0.0, 0.05, 16)
    in_put = []
    for i in range(N):
        in_put.append((taus, N_samples, int_samples))
    in_put = tuple(in_put)
    pool = mp.Pool(processes=N)
    result = pool.map_async(calc_monte_acf, in_put)
    stats = result.get()
    pool.close()
    pool.join()
    
 
    hf = h5py.File('data/monte_cov_iter_{}_Nsamples_{}_intsamples_{}.h5'.format(N, N_samples, int_samples), 'w')
    for i in range(N):
        hf.create_dataset('covariance_{}'.format(i), data = stats[i][0])
    hf.close()
    
    # hf = h5py.File('data/monte_cov_cucu_iter_{}_Nsamples_{}_intsamples_{}.h5'.format(N, N_samples, int_samples), 'w')
    # for i in range(N):
    #     hf.create_dataset('covariance_{}'.format(i), data = stats[i][1])
    # hf.close()
    
    # # hf = h5py.File('data/monte_cov_cus_iter_{}_Nsamples_{}_intsamples_{}.h5'.format(N, N_samples, int_samples), 'w')
    # # for i in range(N):
    # #     hf.create_dataset('covariance_{}'.format(i), data = stats[i][2])
    # # hf.close()
    
    # hf = h5py.File('data/monte_cov_scu_iter_{}_Nsamples_{}_intsamples_{}.h5'.format(N, N_samples, int_samples), 'w')
    # for i in range(N):
    #     hf.create_dataset('covariance_{}'.format(i), data = stats[i][2])
    # hf.close()
    
    # hf = h5py.File('data/monte_cov_ss_iter_{}_Nsamples_{}_intsamples_{}.h5'.format(N, N_samples, int_samples), 'w')
    # for i in range(N):
    #     hf.create_dataset('covariance_{}'.format(i), data = stats[i][3])
    # hf.close()
    
    # # hf = h5py.File('data/monte_mean_til_cov_sensor_iter_{}_Nsamples_{}_intsamples_{}.h5'.format(N, N_samples, int_samples), 'w')
    # # for i in range(N):
    # #     hf.create_dataset('mean_{}'.format(i), data = stats[i][1])
    # # hf.close()

# if __name__=="__main__":
#     N_samples = 10
#     int_samples = 10
    
#     taus = np.linspace(0, 0.01,6)
    
#     in_put = (taus, N_samples, int_samples)
#     a = calc_monte_acf(in_put)