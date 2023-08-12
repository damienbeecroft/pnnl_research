#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 16:24:15 2019

@author: howa549
"""

import jax.numpy as np
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
    
    Tmax = 22
    # pend_dir = "pend_dd"
    pend_dirs = ["pend", "pend_dd", "pend_causal_dd"]

    u = np.asarray([1.0, 1.0])
    u = jax.device_put(u)

    # Output sensor locations and measurements
    y = np.linspace(0, 50, 2000)
    s = odeint(system, u.flatten(), y)

    fig3, bx = plt.subplots(figsize=(4,3))
    for pend_dir in pend_dirs:
        outdir = "C:/Users/beec613/Desktop/pnnl_research/code/damiens_code/good_code/" + pend_dir + "/out_results/pend_0_" + str(Tmax) + "/"
        suff = "MF_loop"

        net_data_dirA = outdir + "results_A/" + suff + "/"


        drange = [0, 50]
        
        d_vx = scipy.io.loadmat(net_data_dirA + "beta_test.mat")
        uA, predA= (d_vx["U_res"].astype(np.float32), d_vx["S_pred"].astype(np.float32))

        # d_vx = scipy.io.loadmat("C:/Users/beec613/Desktop/pnnl_research/code/damiens_code/good_code/" + pend_dir + "/data.mat")
        # t_data, s_data = (d_vx["u"].astype(np.float32), 
        #             d_vx["s"].astype(np.float32))
        
            
        # fig1, ax = plt.subplots(figsize=(8,3))
        # fig2, gx = plt.subplots(figsize=(8,3))
        # fig3, lx = plt.subplots(figsize=(8,3))
        
        # plt.figure(fig2.number)
        # plt.plot(y, s[:, 1], 'k', linestyle=':', linewidth=2, label='Exact')
        # #  plt.scatter(t_data, s_data[:, 1], c='#59a14f')
        # i=1
        # plt.plot(uA[:, 0], predA[:, i],  'k', linestyle='-', label='A prediction')
        err = np.linalg.norm(s[0:400, :].flatten()-predA[0:400, :].flatten(), 2)/np.linalg.norm(s[0:400, :].flatten(), 2)
        errors[0] = err
        print(err)
        
        for i in np.arange(Ntrain):
            lab_str = str(i+1) + " prediction"
            net_data_dir = outdir + "results_" + str(i) + "/" + suff + "/"
            d_vx = scipy.io.loadmat(net_data_dir + "beta_test.mat")
            u, pred= (d_vx["U_res"].astype(np.float32), d_vx["S_pred"].astype(np.float32))
        
            err = np.linalg.norm(s[0:400, :].flatten()-pred[:, 0:400, 0].T.flatten(), 2)/np.linalg.norm(s[0:400, :].flatten(), 2)
            print(err)
            errors[i+1] = err

        plt.figure(fig3.number)

        plt.semilogy(np.arange(Ntrain + 1), errors, marker='o', label = pend_dir)
        plt.xlabel('Level', fontsize=14)
        plt.ylabel('Relative L2 error', fontsize=14)
        plt.tick_params(labelsize=14)
        plt.legend()
        plt.tight_layout()

    plt.savefig('C:/Users/beec613/Desktop/pnnl_research/code/damiens_code/good_code/pend_errors.png', format='png')
    # outdir = "C:/Users/beec613/Desktop/pnnl_research/code/damiens_code/good_code/"
