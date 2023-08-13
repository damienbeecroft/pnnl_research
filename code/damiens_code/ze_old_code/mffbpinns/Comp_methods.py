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

  #  cols_10 <- c("#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
   #              "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC")


import numpy as np
import matplotlib
import math
import matplotlib.pyplot as plt
import scipy.io

if __name__ == "__main__":
    # Modified

    D_vals = np.arange(0, 44)
    C_vals = np.arange(0, 44)
    B_vals = np.arange(0, 44)
    A_vals = np.arange(0, 44)
    save_suff = "_00"
    beta_max_index_test = 20
    N = 10
    error_L2E = np.zeros([4, beta_max_index_test])
    error_L2D = np.zeros([4, beta_max_index_test])
    error_L2C = np.zeros([4, beta_max_index_test])
    error_L2B = np.zeros([4, beta_max_index_test])
    error_L2A = np.zeros([4, beta_max_index_test])


    file='test'
    beta = 19
    for j in np.arange(4):
        save_suff = "_0" + str(j)
        if j == 0 or j ==1:
            N = 10
        else:
            N = 12
        net_data_dirA = "results_A/" + str(10) + "_" + str(len(A_vals))
        net_data_dirB = "results_B/" + str(12) + "_" + str(len(B_vals))
        net_data_dirC = "results_C/" + str(N) + "_" + str(len(C_vals))+ save_suff
        net_data_dirD = "results_D/" + str(N) + "_" + str(len(D_vals))+ save_suff
        net_data_dirE = "results_E/" + str(N) + "_" + str(len(D_vals))+ save_suff
    
        
        data_dir = net_data_dirD + "/beta_"
        d_vx = scipy.io.loadmat(data_dir + file + "_full.mat")
        uHF, xHF, s_xHF, predHF = (d_vx["U_test_h"].astype(np.float32), 
                   d_vx["X_test_h"].astype(np.float32), 
                    d_vx["S_test_h"].astype(np.float32),
                    d_vx["S_pred"].astype(np.float32) )
        

        data_dir = net_data_dirA + "/beta_"
        d_vx = scipy.io.loadmat(data_dir + file + "_full.mat")
        xA, predA= (d_vx["X_test_h"].astype(np.float32), 
                    d_vx["S_pred"].astype(np.float32) )
    
        data_dir = net_data_dirB + "/beta_"
        d_vx = scipy.io.loadmat(data_dir + file + "_full.mat")
        xB, predB= (d_vx["X_test_h"].astype(np.float32), 
                    d_vx["S_pred"].astype(np.float32) )
        
    
        data_dir = net_data_dirC + "/beta_"
        d_vx = scipy.io.loadmat(data_dir + file + "_full.mat")
        xC, predC= (d_vx["X_test_h"].astype(np.float32), 
                    d_vx["S_pred"].astype(np.float32) )
    
    
        data_dir = net_data_dirE + "/beta_"
        d_vx = scipy.io.loadmat(data_dir + file + "_full.mat")
        xE, predE= (d_vx["X_test_h"].astype(np.float32), 
                    d_vx["S_pred"].astype(np.float32) )
    
    
        for i in np.arange(beta_max_index_test):
            error_L2E[j, i] = np.linalg.norm(s_xHF[i, :, :]-predE[i, :, :], 2) / np.linalg.norm(s_xHF[i, :, :], 2) 
    
        for i in np.arange(beta_max_index_test):
            error_L2D[j, i] = np.linalg.norm(s_xHF[i, :, :]-predHF[i, :, :], 2) / np.linalg.norm(s_xHF[i, :, :], 2) 
    
        
        for i in np.arange(beta_max_index_test):
            error_L2C[j, i] = np.linalg.norm(s_xHF[i, :, :]-predC[i, :, :], 2) / np.linalg.norm(s_xHF[i, :, :], 2) 
    
        for i in np.arange(beta_max_index_test):
            error_L2B[j, i] = np.linalg.norm(s_xHF[i, :, :]-predB[i, :, :], 2) / np.linalg.norm(s_xHF[i, :, :], 2) 
    
        
        for i in np.arange(beta_max_index_test):
            error_L2A[j, i] = np.linalg.norm(s_xHF[i, :, :]-predA[i, :, :], 2) / np.linalg.norm(s_xHF[i, :, :], 2) 
    
    
    
    e1vec = np.arange(0, 4)
    e2vec = np.arange(4, 8)
    e3vec = np.arange(8, 12)
    e4vec = np.arange(12, 16)
    e5vec = np.arange(16, 20)

    eE  = [np.mean(error_L2E[:, e1vec], axis=1),
          np.mean(error_L2E[:, e2vec], axis=1),
          np.mean(error_L2E[:, e3vec], axis=1),
          np.mean(error_L2E[:, e4vec], axis=1),
          np.mean(error_L2E[:, e5vec], axis=1)]

    eE = np.asarray(eE).T
    

    evec  = np.arange(5)


    fig3, gx = plt.subplots(figsize=(7, 4))
    plt.figure(fig3.number)
    plt.plot(evec, eE[0, :], '#59a14f', marker='s', linestyle=':', label='00 error')
    plt.plot(evec, eE[1, :], '#4e79a7', marker='o', linestyle='-.', label='01 error')
    plt.plot(evec, eE[2, :], '#e15759', marker='^', linestyle='--', label='02 error')
    plt.plot(evec, eE[3, :], '#F28E2B', marker='v', linestyle='-', label='03 error')

    plt.legend(fontsize=12)
    labels = ['A', 'B', 'C', 'D', 'E']
    plt.xticks(evec, labels)


    plt.ylabel('Relative $L_2$ error from E', fontsize=16)
    plt.tick_params(labelsize=16)
    plt.tight_layout()

    plt.savefig('Error_comp_methods.eps', format='eps')

    fig3, gx = plt.subplots(figsize=(7, 4))
    plt.figure(fig3.number)
    evec  = np.linspace(14, 20, 20)

    plt.plot(evec, error_L2E[0, :], '#59a14f', marker='s', linestyle=':', label='00 error')
    plt.plot(evec, error_L2E[1, :], '#4e79a7', marker='o', linestyle='-.', label='01 error')
    plt.plot(evec, error_L2E[2, :], '#e15759', marker='^', linestyle='--', label='02 error')
    plt.plot(evec, error_L2E[3, :], '#F28E2B', marker='v', linestyle='-', label='03 error')

    plt.legend(fontsize=12)


    plt.ylabel('Relative $L_2$ error from E', fontsize=16)
    plt.tick_params(labelsize=16)
    plt.tight_layout()

    plt.savefig('Error_comp_methods2.eps', format='eps')
    
    fig3, gx = plt.subplots(figsize=(7, 4))
    plt.figure(fig3.number)
    evec  = np.linspace(14, 20, 20)

    plt.plot(evec, error_L2D[0, :], '#59a14f', marker='s', linestyle=':', label='00 error')
    plt.plot(evec, error_L2D[1, :], '#4e79a7', marker='o', linestyle='-.', label='01 error')
    plt.plot(evec, error_L2D[2, :], '#e15759', marker='^', linestyle='--', label='02 error')
    plt.plot(evec, error_L2D[3, :], '#F28E2B', marker='v', linestyle='-', label='03 error')

    plt.legend(fontsize=12)


    plt.ylabel('Relative $L_2$ error from D', fontsize=16)
    plt.tick_params(labelsize=16)
    plt.tight_layout()

    plt.savefig('Error_comp_methods2D.eps', format='eps')
        