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
import matplotlib.pyplot as plt
import scipy.io
from jax import random
from jax.experimental.ode import odeint
import jax

if __name__ == "__main__":
    # Modified



    dim_HF_branch = 1
    dim_x_trunk = 1     # (x,y)
    cur_data_dir  = 'results_4/MF_loop_2/'
    net_data_dirHF = cur_data_dir 
    xmax = 1
    xmin = 0
    
    lines = False
    l1 = .0
    l2 = .2
    
    data_dir = net_data_dirHF + "beta_"
    d_vx = scipy.io.loadmat(data_dir + "test.mat")
    t,  x, U_pred, U_star= (d_vx["t"].astype(np.float32), 
                d_vx["x"].astype(np.float32),
                d_vx["U_pred"].astype(np.float32),
                d_vx["U_star"].astype(np.float32))


        
        
    plt.figure(figsize=(10, 3))
    plt.subplot(1, 3, 1)
    plt.pcolor(t, x, U_star, cmap='jet', shading='auto', vmax=1.5, vmin=-1.5)
    plt.xlim([xmin, xmax])

    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=14) 

    plt.xlabel('$t$', fontsize=14)
    plt.ylabel('$x$', fontsize=14)
    plt.title('Exact u(t, x)', fontsize=14)
    plt.tick_params(labelsize=14)

    plt.tight_layout()
    
    plt.subplot(1, 3, 2)
    plt.pcolor(t, x, U_pred, cmap='jet', shading='auto', vmax=1.5, vmin=-1.5)
    plt.xlim([xmin, xmax])
    cbar = plt.colorbar()
    if lines == True:
        plt.plot([l1, l1], [0, 1], 'k', linewidth=2)
        plt.plot([l2, l2], [0, 1], 'k', linewidth=2)
    plt.xlabel('$t$', fontsize=14)
    plt.ylabel('$x$', fontsize=14)
    plt.title('Predicted u(t, x)', fontsize=14)
    plt.tick_params(labelsize=14)
    cbar.ax.tick_params(labelsize=14) 

    plt.tight_layout()
    
    plt.subplot(1, 3, 3)
    plt.pcolor(t, x, np.abs(U_star - U_pred), cmap='jet', shading='auto')
    if lines == True:
        plt.plot([l1, l1], [0, 1], 'w', linewidth=2)
        plt.plot([l2, l2], [0, 1], 'w', linewidth=2)
    plt.xlim([xmin, xmax])

    cbar = plt.colorbar()
    plt.xlabel('$t$', fontsize=14)
    plt.ylabel('$x$', fontsize=14)
    plt.title('Absolute error', fontsize=14)
    plt.tick_params(labelsize=14)
    cbar.ax.tick_params(labelsize=14) 

    plt.tight_layout()

    plt.savefig(cur_data_dir + '/A.png', format='png')
    
    error_u = np.linalg.norm(U_star - U_pred, 2) / np.linalg.norm(U_star, 2)

    print('Relative L2 error_u: %e' % (error_u))
    
    
    fig1, ax = plt.subplots()
    plt.figure(fig1.number)   
    d_vx = scipy.io.loadmat(net_data_dirHF +'/losses.mat')
    train, res, ics, data = ( d_vx["training_loss"].astype(np.float32),
             d_vx["res_loss"].astype(np.float32),
             d_vx["ics_loss"].astype(np.float32),
             d_vx["ut_loss"].astype(np.float32))
    
    step = np.arange(0, 1000*len(train[0]), 1000)
    plt.semilogy(step, train[0], 'k', linestyle='--', label='Training loss')
    plt.semilogy(step, data[0], '#4e79a7', linestyle=':', label='ut loss A')
    plt.semilogy(step, res[0], '#59a14f', linestyle='--', label='Res loss A')
    plt.semilogy(step, ics[0], '#e15759', linestyle='--', label='ICS loss A')

    
    plt.xlabel('Number of Epochs', fontsize=20)
    plt.ylabel(r'Loss', fontsize=20)
    plt.legend(fontsize=12)
    
    plt.tick_params(labelsize=16)
    plt.tight_layout()
    plt.savefig(cur_data_dir + '/Loss.png', format='png')
    
