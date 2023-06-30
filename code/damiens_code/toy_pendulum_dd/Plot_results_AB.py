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
    
    b = 0.05
    g = 9.81
    l = 1
    m = 1
    
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
    
    

    suff = "MF_testMF"
    save_suff = "_l_0"

    net_data_dirA = "results_A/" + suff + "/"
    net_data_dirB = "results_B/" + suff + "/"
    net_data_dirC = "results_C/" + suff + save_suff+ "/"

    net_data_dirD = "results_D/" + suff + save_suff+ "/"
    net_data_dirE = "results_E/" + suff + save_suff+ "/"
    net_data_dirF = "results_F/" + suff + save_suff+ "/"

    drange = [0, 50]
    
    d_vx = scipy.io.loadmat(net_data_dirA + "beta_test.mat")
    uA, predA= (d_vx["U_res"].astype(np.float32), d_vx["S_pred"].astype(np.float32))

 
    d_vx = scipy.io.loadmat("data.mat")
    t_data, s_data = (d_vx["u"].astype(np.float32), 
               d_vx["s"].astype(np.float32))
  
        
    fig3, gx = plt.subplots(figsize=(8,3))
    plt.figure(fig3.number)
    plt.plot(y, s[:, 0], 'k', linestyle=':', linewidth=2, label='exact')
    i=0
    plt.plot(uA[:, i], predA[:, i],  '#59a14f', linestyle='-', label='A prediction')
   # plt.plot(uB[:, i], predB[:, i],  '#4e79a7', linestyle='-.', label='B prediction')

  #  plt.plot(uE[:, i], predE[:, i], '#BAB0AC', linestyle='--', label='E prediction')
 #   plt.plot(uF[:, i], predF[:, i], '#4e79a7', linestyle='-', label='F prediction')
    plt.xlim(drange)
    plt.plot([4, 4], [-1.5, 1.5], 'k')
    plt.plot([8, 8], [-1.5, 1.5], 'k')
    plt.plot([12, 12], [-1.5, 1.5], 'k')
    plt.plot([16, 16], [-1.5, 1.5], 'k')
  #  plt.legend(fontsize=12, ncol=3)
    plt.ylim([-1.5, 1.5])
    plt.xlim([-0, 10])
    plt.tick_params(labelsize=16)
    plt.tight_layout()
    plt.savefig(suff + save_suff + '_s1.png', format='png')

    fig3, gx = plt.subplots(figsize=(8,3))
    plt.figure(fig3.number)
    plt.plot(y, s[:, 1], 'k', linestyle=':', linewidth=2, label='exact')
    i=1
    plt.plot(uA[:, 0], predA[:, i],  '#59a14f', linestyle='-', label='A prediction')
   # plt.plot(uB[:, 0], predB[:, i],  '#4e79a7', linestyle='-.', label='B prediction')

  #  plt.plot(uE[:, 0], predE[:, i], '#BAB0AC', linestyle='--', label='E prediction')
  #  plt.plot(uF[:, 0], predF[:, i], '#4e79a7', linestyle='-', label='F prediction')

    plt.xlim(drange)
    plt.plot([4, 4], [-3.5, 3.5], 'k')
    plt.plot([8, 8], [-3.5, 3.5], 'k')
    plt.plot([12, 12], [-3.5, 3.5], 'k')
    plt.plot([16, 16], [-3.5, 3.5], 'k')
 #   plt.legend(fontsize=12, ncol=3)
    plt.tick_params(labelsize=16)
    plt.tight_layout()
    plt.ylim([-3.5, 3.5])
    plt.xlim([-0, 10])


    plt.savefig( suff + save_suff + '_s2.png', format='png')

    
