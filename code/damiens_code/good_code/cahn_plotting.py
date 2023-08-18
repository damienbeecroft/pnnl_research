#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 16:24:15 2019

@author: howa549
"""

#  
 # CS1 = plt.contour(X, Y, phi1.transpose(), levels=[0.5], colors=('#59a14f'), linestyles=('--'), linewidths=(2))
 # CS2 = plt.contour(X, Y, phi2.transpose(), levels=[0.5], colors=('#4e79a7'), linestyles=('-.'), linewidths=(2))
 # CS3 = plt.contour(X, Y, phi3.transpose(), levels=[0.5], colors=('#e15759'), linestyles=(':'), linewidths=(2))
 # CS4 = plt.contour(X, Y, phi4.transpose(), levels=[0.5], colors=('k'), linestyles=('-'), linewidths=(1.5))



import jax.numpy as np
import matplotlib
import math
import numpy as onp
import matplotlib.pyplot as plt
import scipy.io
from jax import random
from jax.experimental.ode import odeint
import jax
import matplotlib.colors as colors
if __name__ == "__main__":
    # Modified

    n_runs = 3
    
    errors = onp.zeros(n_runs + 1)

    # Tmaxes = [1.0]
    # learning_rates = [0.01,0.001,0.0001]
    # decay_rates = [0.95,0.99]
    # widths = [30,40]

    Tmaxes = [1.0]
    learning_rates = [0.01]
    decay_rates = [0.95]
    widths = [30]
    for Tmax in Tmaxes:
        for learning_rate in learning_rates:
            for decay_rate in decay_rates:
                for width in widths:
                    path = 'C:/Users/beec613/Desktop/pnnl_research/code/damiens_code/good_code/allen_cahn_dd/saved_results2/run2/cahn_0_' + str(Tmax) + '_' + str(learning_rate) + '_' + str(decay_rate) + '_' + str(width) + '/'
                    # A
                    fig1, ax = plt.subplots()

                    post = 'MF_loop_res10/'
                    net_data_dirHF  = path + 'results_A/' + post
                    xmax = 1
                    xmin = 0

                    
                    data_dir = net_data_dirHF + "beta_"
                    d_vx = scipy.io.loadmat(data_dir + "test.mat")
                    t,  x, U_pred, U_star= (d_vx["t"].astype(np.float32), 
                                d_vx["x"].astype(np.float32),
                                d_vx["U_pred"].astype(np.float32),
                                d_vx["U_star"].astype(np.float32))

                    
                    error_u = np.linalg.norm(U_star - U_pred, 2) / np.linalg.norm(U_star, 2)
                    errors[0] = error_u
                    
                    
                    for i in np.arange(n_runs):
                        net_data_dirHF  = path + 'results_' + str(i) +"/" +  post
                        data_dir = net_data_dirHF + "beta_"
                        d_vx = scipy.io.loadmat(data_dir + "test.mat")
                        t,  x, U_pred= (d_vx["t"].astype(np.float32), 
                                d_vx["x"].astype(np.float32),
                                d_vx["U_pred"].astype(np.float32))
                        
                        error_u = np.linalg.norm(U_star - U_pred, 2) / np.linalg.norm(U_star, 2)

                        print('Relative L2 error_u: %e' % (error_u))
                        
                        errors[i+1] = error_u
                        

    n_runs2 = 3
    
    errors2 = onp.zeros(n_runs2 + 1)

    plt.figure(figsize=(5, 4))
    plt.semilogy(np.arange(n_runs2 + 1), errors2, marker='o')
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Relative $L_2$ error', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.tight_layout()
    path = 'C:/Users/beec613/Desktop/pnnl_research/code/damiens_code/good_code/allen_cahn/saved_results2/run2/cahn_0_1.0_0.001_0.95_30/'
    # A
    # fig1, ax = plt.subplots()

    post = 'MF_loop_res10/'
    net_data_dirHF  = path + 'results_A/' + post
    xmax = 1
    xmin = 0

    
    data_dir = net_data_dirHF + "beta_"
    d_vx = scipy.io.loadmat(data_dir + "test.mat")
    t,  x, U_pred, U_star= (d_vx["t"].astype(np.float32), 
                d_vx["x"].astype(np.float32),
                d_vx["U_pred"].astype(np.float32),
                d_vx["U_star"].astype(np.float32))

    
    error_u = np.linalg.norm(U_star - U_pred, 2) / np.linalg.norm(U_star, 2)
    errors2[0] = error_u
    
    
    for i in np.arange(n_runs2):
        net_data_dirHF  = path + 'results_' + str(i) +"/" +  post
        data_dir = net_data_dirHF + "beta_"
        d_vx = scipy.io.loadmat(data_dir + "test.mat")
        t,  x, U_pred= (d_vx["t"].astype(np.float32), 
                d_vx["x"].astype(np.float32),
                d_vx["U_pred"].astype(np.float32))
        
        error_u = np.linalg.norm(U_star - U_pred, 2) / np.linalg.norm(U_star, 2)

        print('Relative L2 error_u: %e' % (error_u))
        
        errors2[i+1] = error_u
        
    
    plt.figure(figsize=(5, 4))
    plt.semilogy(np.arange(n_runs + 1), errors, marker='o', label='MF Allen-Cahn w/ DD')
    plt.semilogy(np.arange(n_runs2 + 1), errors2, marker='o', label='MF Allen-Cahn')
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Relative $L_2$ error', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.legend()
    plt.tight_layout()
    

    plt.savefig('C:/Users/beec613/Desktop/pnnl_research/code/damiens_code/good_code/cahn_errors.png', format='png')

                

    
    
    