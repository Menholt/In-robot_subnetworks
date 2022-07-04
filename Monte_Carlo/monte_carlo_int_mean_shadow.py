# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 10:07:54 2022

@author: EG
"""
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import h5py

def monte_int_rej_both(f_cu, f_s, height, width, n_nodes, N_samples, min_dist, r, L_samples, N_sg, sigma, delta, stepsize):
    # area = (height*width)**n_nodes
    summ = 0
    for i in range(N_samples):
        grf = createMap(width, height, sigma, delta, stepsize)
        samples = np.zeros((2, n_nodes))
        for j in range(n_nodes):
            samples[0, j] = np.random.uniform(-width/2 + r, width/2 - r)
            samples[1, j] = np.random.uniform(-height/2 + r, height/2 - r)
            k = 0
            while k < j:
                if np.linalg.norm(samples[:,j]-samples[:,k])<min_dist:
                    samples[0, j] = np.random.uniform(-width/2 + r, width/2 - r)
                    samples[1, j] = np.random.uniform(-height/2 + r, height/2 - r)
                    k = 0
                else:
                    k += 1
        summ += f_cu(samples, height, width, min_dist, grf, delta, stepsize)
        summ += f_s(samples, height, width, min_dist, r, L_samples, grf, delta, stepsize)
    return summ/N_samples*((N_sg-1)/(2*n_nodes-2))


def monte_int_rej_sensor(f_s, height, width, n_nodes, N_samples, min_dist, r, L_samples, N_sg, sigma, delta, stepsize):
    # area = (height*width)**n_nodes
    summ = 0
    for i in range(N_samples):
        grf = createMap(width, height, sigma, delta, stepsize)
        samples = np.zeros((2, n_nodes))
        for j in range(n_nodes):
            samples[0, j] = np.random.uniform(-width/2 + r, width/2 - r)
            samples[1, j] = np.random.uniform(-height/2 + r, height/2 - r)
            k = 0
            while k < j:
                if np.linalg.norm(samples[:,j]-samples[:,k])<min_dist:
                    samples[0, j] = np.random.uniform(-width/2 + r, width/2 - r)
                    samples[1, j] = np.random.uniform(-height/2 + r, height/2 - r)
                    k = 0
                else:
                    k += 1
        summ += f_s(samples, height, width, min_dist, r, L_samples, grf, delta, stepsize)
    return summ/N_samples*((N_sg-1)/(n_nodes-2))


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
    return 10**(((1-np.exp(-dist/delta))/(np.sqrt(2)*np.sqrt(1+np.exp(-dist/delta)))*(grf1+grf2))/10)


def pathloss(x):
    return min(1,np.linalg.norm(x)**(-3))

def f_cu_receiver_cu(x, height, width, min_dist, grf, delta, stepsize):
    pathlosses = [pathloss(x[:,n]-x[:,0]) for n in range(1, len(x[0,:]))]
    shadows = np.array([shadowing(grf, delta, x[:,n], x[:,0], stepsize, height, width) for n in range(1, len(x[0,:]))])
    output = np.sum(np.multiply(pathlosses, shadows))
    return output


def f_s_receiver_cu(x, height, width, min_dist, r, N_int, grf, delta, stepsize):
    sum_of_pathlosses_shadows = 0
    
    for j in range(N_int):
        xi = np.zeros_like(x)
        k=0
        while k < len(x[0,:]):
            displacement = np.random.uniform(-r, r, 2)
            if np.linalg.norm(displacement)<=r:
                xi[:,k] = x[:,k]+displacement
                k+=1
        pathlosses = np.array([pathloss(xi[:,n]-x[:,0]) for n in range(1, len(x[0,:]))])
        shadows = np.array([shadowing(grf, delta, xi[:,n], x[:,0], stepsize, height, width) for n in range(1, len(x[0,:]))])
        
        sum_of_pathlosses_shadows += np.sum(np.multiply(pathlosses, shadows))
        
    output =  sum_of_pathlosses_shadows/N_int
    return output

def calc_monte_mean(in_put):
    np.random.seed()
    N_iter, N_samples, L_samples, sensor = in_put
    N_iter = int(N_iter)
    height = 30
    width = 30
    n_nodes = 16
    min_dist = 3
    r = 2
    N_sg = 4
    delta = 5
    sigma = 3
    stepsize = 1/20
    monte_mean = np.zeros(N_iter)
    if sensor:
        for j in tqdm(range(N_iter)):
            monte_mean[j] = (monte_int_rej_sensor(f_s_receiver_cu, height, width, n_nodes, N_samples, min_dist, r, L_samples, N_sg, sigma, delta, stepsize))
    else:
        for j in tqdm(range(N_iter)):
            monte_mean[j] = (monte_int_rej_both(f_cu_receiver_cu, f_s_receiver_cu, height, width, n_nodes, N_samples, min_dist, r, L_samples, N_sg, sigma, delta, stepsize))
    return monte_mean


if __name__ == "__main__":
    N = mp.cpu_count()
    N_iter = int(1024/N)
    N_sampless = [50, 25, 10]
    L_sampless = [25, 10]
    N_samples = 100
    L_samples = 25
    for N_samples in N_sampless:
        for L_samples in L_sampless:
            sensor = False
            in_put = []
            
            for i in range(N):
                in_put.append((N_iter, N_samples, L_samples, sensor))
            in_put = tuple(in_put)
            pool = mp.Pool(processes=N)
            result = pool.map_async(calc_monte_mean, in_put)
            pool.close()
            pool.join()
            stats = result.get()
            if sensor:
                hf = h5py.File('data/monte_mean_sensor_iter_{}_Nsamples_{}_Lsamples_{}.h5'.format(N_iter*N, N_samples, L_samples), 'w')
                for i in range(N):
                    hf.create_dataset('monte_means_{}'.format(i), data = stats[i])
                hf.close()
            else:
                hf = h5py.File('data/monte_mean_iter_{}_Nsamples_{}_Lsamples_{}.h5'.format(N_iter*N, N_samples, L_samples), 'w')
                for i in range(N):
                    hf.create_dataset('monte_means_{}'.format(i), data = stats[i])
                hf.close()

# if __name__ == '__main__':
#     N_iter = 10
#     a = calc_monte_mean(N_iter)
