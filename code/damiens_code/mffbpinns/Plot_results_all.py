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
import numpy as onp

if __name__ == "__main__":
    # Modified
    
    b = 0.05
    g = 9.81
    l = 1
    m = 1
    Ntrain = 3
    
    errors = onp.zeros([Ntrain+1])
    
    def system(s, t):
      s1 = s[0]
      s2 = s[1]
      ds1_dt = s2
      ds2_dt =  - (b/m) * s2 - g * np.sin(s1)
      ds_dt = [ds1_dt, ds2_dt]
      return ds_dt

    u = np.asarray([1.0, 1.0])
    u = jax.device_put(u)

    # Output sensor locations and measurements
    y = np.linspace(0, 50, 2000)
    s = odeint(system, u.flatten(), y)
    
    
    outdir = ""
    suff = "MF_loop"

    net_data_dirA = outdir + "results_A/" + suff + "/"


    drange = [0, 50]
    
    d_vx = scipy.io.loadmat(net_data_dirA + "beta_test.mat")
    uA, predA= (d_vx["U_res"].astype(np.float32), d_vx["S_pred"].astype(np.float32))


    d_vx = scipy.io.loadmat("data.mat")
    t_data, s_data = (d_vx["u"].astype(np.float32), 
               d_vx["s"].astype(np.float32))
  
        
    fig1, ax = plt.subplots(figsize=(8,3))
    fig2, gx = plt.subplots(figsize=(8,3))

    plt.figure(fig1.number)
    plt.plot(y, s[:, 0], 'k', linestyle=':', linewidth=2, label='Exact')
 #   plt.scatter(t_data, s_data[:, 0], c='#59a14f')
    i=0
    plt.plot(uA[:, i], predA[:, i],  'k', linestyle='-', label='A prediction')
    
    
    
    plt.figure(fig2.number)
    plt.plot(y, s[:, 1], 'k', linestyle=':', linewidth=2, label='Exact')
  #  plt.scatter(t_data, s_data[:, 1], c='#59a14f')
    i=1
    plt.plot(uA[:, 0], predA[:, i],  'k', linestyle='-', label='A prediction')
    err = np.linalg.norm(s[0:400, :].flatten()-predA[0:400, :].flatten(), 2)/np.linalg.norm(s[0:400, :].flatten(), 2)
    errors[0] = err
    print(err)
    
    
    
    for i in np.arange(Ntrain):
        lab_str = str(i+1) + " prediction"
        net_data_dir = outdir + "results_" + str(i) + "/" + suff + "/"
        d_vx = scipy.io.loadmat(net_data_dir + "beta_test.mat")
        u, pred= (d_vx["U_res"].astype(np.float32), d_vx["S_pred"].astype(np.float32))
    
        plt.figure(fig1.number)
     #   plt.scatter(t_data, s_data[:, 0], c='#59a14f')
        plt.plot(u[:, 0], pred[0, :, 0],  label = lab_str)
        
        
        
        plt.figure(fig2.number)
      #  plt.scatter(t_data, s_data[:, 1], c='#59a14f')
        plt.plot(u[:, 0], pred[1, :, 0],   label = lab_str)
        
        err = np.linalg.norm(s[0:400, :].flatten()-pred[:, 0:400, 0].T.flatten(), 2)/np.linalg.norm(s[0:400, :].flatten(), 2)
        print(err)
        errors[i+1] = err
        
    plt.figure(fig1.number)

    plt.xlim(drange)
    plt.legend(fontsize=12, ncol=1, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylim([-1.5, 1.5])
    plt.xlim([-0, 10])
    plt.tick_params(labelsize=16)
    plt.tight_layout()
    plt.savefig(suff +  '_s1.png', format='png')

    plt.figure(fig2.number)
  #  plt.scatter(t_data, s_data[:, 1], c='#59a14f')

    plt.xlim(drange)

    plt.legend(fontsize=12, ncol=1, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tick_params(labelsize=16)
    plt.tight_layout()
    plt.ylim([-3.5, 3.5])
    plt.xlim([-0, 10])


    plt.savefig( suff  + '_s2.png', format='png')

    fig3,bx = plt.subplots(figsize=(4,3))
    plt.figure(fig3.number)

    plt.semilogy(np.arange(Ntrain + 1), errors, marker='o')
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Relative L2 error', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.tight_layout()

    plt.savefig(suff + '_Errors.png', format='png')
