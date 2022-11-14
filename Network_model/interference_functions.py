# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 14:42:37 2022

@author: EG
"""
import numpy as np
# from tqdm import tqdm
from numba import njit


def createMap(width, height, sigmaS, correlationDistance, stepsize):
    num_x_points = int(width/stepsize) + 3
    num_y_points = int(height/stepsize) + 3
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
    grf1 = grf[int((loc1[0]+width/2)/stepsize)+1][int((loc1[1]+height/2)/stepsize)+1]
    grf2 = grf[int((loc2[0]+width/2)/stepsize)+1][int((loc2[1]+height/2)/stepsize)+1]
    dist = np.linalg.norm(loc1-loc2)
    return 10**(((1-np.exp(-dist/delta))/(np.sqrt(2)*np.sqrt(1+np.exp(-dist/delta)))*(grf1+grf2))/10)


def auto_correlation(x, max_predict):
    auto_corr = np.zeros(max_predict)
    auto_corr[0] = np.corrcoef(x)
    for i in range(1, max_predict):
        auto_corr[i] = np.corrcoef(x[i:],x[:-i])[0,1]*(len(x)-i)/len(x)
    return auto_corr

def subnetwork_layout(M, r, x, direction):
    angles = np.linspace(0, 2*np.pi, M, endpoint = False) + direction
    pairs = np.zeros((M, 2))
    for i in range(M):
        pairs[i,0] = x[0] + np.cos(angles[i]) * r
        pairs[i,1] = x[1] + np.sin(angles[i]) * r
    return pairs

def subnetwork_layout_random(M, r, x, direction):
    angles = np.linspace(0, 2*np.pi, M, endpoint = False) + direction
    pairs = np.zeros((M, 2))
    for i in range(M):
        pairs[i,0] = x[0] + np.cos(angles[i]) * r
        pairs[i,1] = x[1] + np.sin(angles[i]) * r
    return pairs

def interfere_location(x, direction, displacement):
    angle = displacement[0] + direction
    pair = np.zeros(2)
    
    pair[0] = x[0] + np.cos(angle)* displacement[1]
    pair[1] = x[1] + np.sin(angle) * displacement[1]
    return pair

def interfere_location_no_rotate(x, direction, displacement):
    pair = np.zeros(2)
    
    pair[0] = x[0] + displacement[1]
    pair[1] = x[1] + displacement[1]
    return pair

def interfere_displacement(cell_radius):
    theta = np.random.uniform(0, 2*np.pi)
    radius = np.sqrt(np.random.uniform(0,1))*cell_radius
    return [theta, radius]

def collision_nodes(x_0, x_1, spread = np.pi):
    # if x_0[0] != x_1[0]:
    tan_ang = np.arctan((x_0[1]-x_1[1]) / (x_0[0]-x_1[0]))
    if x_0[0] > x_1[0]:
        dir_x0 = np.random.uniform(tan_ang - spread/2, tan_ang + spread/2)
        dir_x1 = np.random.uniform(tan_ang + spread/2 + np.pi, tan_ang + spread/2 + np.pi)
    else:
        dir_x0 = np.random.uniform(tan_ang + spread/2 + np.pi, tan_ang + spread/2 + np.pi)
        dir_x1 = np.random.uniform(tan_ang - spread/2, tan_ang + spread/2)
    return dir_x0, dir_x1


def collision_nodes_random(x_0, x_1):
    dir_x0 = np.random.uniform(0,2*np.pi)
    dir_x1 = np.random.uniform(0,2*np.pi)
    return dir_x0, dir_x1


def deployment(n_points, width, height, min_dist, cell_radius):
    
    x_supp = [-width/2 + cell_radius, width/2 - cell_radius]
    y_supp = [-height/2 + cell_radius, height/2 - cell_radius]
    node_loc = np.zeros((n_points,2))
    counters = np.zeros(n_points)
    
    for i in range(n_points):
        node_loc[i,0] = np.random.uniform(x_supp[0], x_supp[1])
        node_loc[i,1] = np.random.uniform(y_supp[0], y_supp[1])
        
        #Check cell radius and minimum distance
        for j in range(i):
            if (np.linalg.norm(node_loc[i]-node_loc[j]) < min_dist):
                suitable = np.zeros(i)
                while sum(suitable) < i:
                    node_loc[i,0] = np.random.uniform(x_supp[0], x_supp[1])
                    node_loc[i,1] = np.random.uniform(y_supp[0], y_supp[1])
                    for m in range(i):
                        if(np.linalg.norm(node_loc[i] - node_loc[m]) < min_dist):
                            suitable[m] = False
                        else:
                            suitable[m] = True
                    counters[i] += 1
    return node_loc, counters


def new_deployment(node_loc, idx, width, height, min_dist, cell_radius):
    """
    Redeploys a node with index idx after being stuck in another node for 0.1 second.
    """
    x_supp = [-width/2 + cell_radius, width/2 - cell_radius]
    y_supp = [-height/2 + cell_radius, height/2 - cell_radius]
    new_node_loc = np.zeros(2)
    
    #Check cell radius and minimum distance
    suitable = np.zeros(len(node_loc))
    while sum(suitable) < len(node_loc):
        suitable = np.zeros(len(node_loc))
        new_node_loc[0] = np.random.uniform(x_supp[0], x_supp[1])
        new_node_loc[1] = np.random.uniform(y_supp[0], y_supp[1])
        for m in range(len(node_loc)):
            if(np.linalg.norm(new_node_loc - node_loc[m]) < min_dist and m != idx):
                suitable[m] = False
            else:
                suitable[m] = True
    direction = np.random.uniform(0, 2*np.pi)
    return new_node_loc, direction

def location_of_transmitter(x_0, direction, distance):
    x = np.zeros_like(x_0)
    x[0] = x_0[0] + np.cos(direction)*distance
    x[0] = x_0[0] + np.sin(direction)*distance
    return x


def random_direction(node_loc, directions, vel, min_dist, cell_radius, time_step, height, width):
    node_loc_upd = np.zeros_like(node_loc)
    directions_upd = np.zeros_like(directions)
    for i in range(len(node_loc)):
        node_loc_upd[i,0] = node_loc[i,0] + np.cos(directions[i]) * vel*time_step 
        node_loc_upd[i,1] = node_loc[i,1] + np.sin(directions[i]) * vel*time_step 

    for i in range(len(node_loc)):   
        collision_counter = 0
        
        #Collision with another node
        for j in range(i):
            if np.linalg.norm(node_loc_upd[i] - node_loc_upd[j]) < min_dist:
                directions_upd[i] = np.random.uniform(0,2*np.pi)
                directions_upd[j] = np.random.uniform(0,2*np.pi)
                # directions_upd[i], directions_upd[j] = collision_nodes_random(node_loc[i], node_loc[j])
                # directions_upd[i], directions_upd[j] = collision_nodes(node_loc[i], node_loc[j])
                collision_counter += 1
                
        if collision_counter == 0:
            directions_upd[i] = directions[i]
         #Collision with wall
        if(node_loc_upd[i,0] < -width/2 + cell_radius):
            directions_upd[i] = np.random.uniform(-np.pi/2+np.pi/8, np.pi/2-np.pi/8)
            # node_loc_upd[i,0] = node_loc[i,0] + 2 * np.cos(directions[i]) * vel*time_step 
        if(node_loc_upd[i,0] > width/2 - cell_radius):
            directions_upd[i] = np.random.uniform(np.pi/2+np.pi/8, 3*np.pi/2-np.pi/8)
            # node_loc_upd[i,0] = node_loc[i,0] + 2 * np.cos(directions[i]) * vel*time_step 
        if(node_loc_upd[i,1] < -height/2 + cell_radius):
            directions_upd[i] = np.random.uniform(0+np.pi/8, np.pi-np.pi/8)
            # node_loc_upd[i,1] = node_loc[i,1] + 2 * np.sin(directions[i]) * vel*time_step 
        if (node_loc_upd[i,1] > height/2 - cell_radius):
            directions_upd[i] = np.random.uniform(-np.pi+np.pi/8, 0-np.pi/8)
            # node_loc_upd[i,1] = node_loc[i,1] + 2 * np.sin(directions[i]) * vel*time_step 
    return node_loc_upd, directions_upd  

# @njit
def path_loss(x_0, x, alpha):
    return min(1, np.linalg.norm(x-x_0)**(-alpha))
    # return np.linalg.norm(x-x_0)**(-alpha)




def small_scale_fading(t, M, doppler, beta_n, theta):
    H = np.abs(np.sqrt(2/M)*(np.sum((np.cos(beta_n)+1j*np.sin(beta_n))*np.cos(doppler*t+theta))))
    return H**2
    # return 1



def traffic_pulses_2(pct_uplink):
    if np.random.uniform()<pct_uplink:
        return 1
    else:
        return 0
    
def interference_power(kappa, x_0, x, alpha, t, theta, M, doppler, beta_n, grf, delta, stepsize, height, width):
    path_l = path_loss(x_0, x, alpha)
    ss_fading = small_scale_fading(t, M, doppler, beta_n, theta)
    shadow = shadowing(grf, delta, x, x_0, stepsize, height, width)
    # return kappa*path_loss(x_0, x, alpha)*small_scale_fading(t,node_number, max_doppler, M, N, K, alpha_n, doppler, beta_n, 0)**2 * shadowing(grf, delta, x, x_0, stepsize, height, width)
    interference = path_l * ss_fading * shadow
    return interference

def interference_power_med_resten(kappa, x_0, x, alpha, t, theta, M, doppler, beta_n, grf, delta, stepsize, height, width):
    path_l = path_loss(x_0, x, alpha)
    ss_fading = small_scale_fading(t, M, doppler, beta_n, theta)
    shadow = shadowing(grf, delta, x, x_0, stepsize, height, width)
    # return kappa*path_loss(x_0, x, alpha)*small_scale_fading(t,node_number, max_doppler, M, N, K, alpha_n, doppler, beta_n, 0)**2 * shadowing(grf, delta, x, x_0, stepsize, height, width)
    interference = path_l * ss_fading * shadow
    return interference, path_l, ss_fading, shadow