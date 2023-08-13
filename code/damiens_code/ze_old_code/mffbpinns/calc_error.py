#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 14:37:41 2023

@author: howa549
"""
import jax.numpy as np
import matplotlib
import math
import matplotlib.pyplot as plt
import scipy.io
from jax import random
from jax.experimental.ode import odeint
import jax

def run_MF(replay, MAS, RDPS, scaled, N, MASl=0):
    
  # suff = "MFpen_newMAS" + str(N)
   suff = "SF_" + str(N)
   if replay:
        suff = suff + "_replay"
   if MAS:
        suff = suff + "_MAS"
   if scaled:
        suff = suff + "scaled"
   if RDPS:
        suff = suff + "_RDPS"
        
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
   
   save_suff = "_l_" + str(MASl)

   net_data_dirF = "results_MAS_new/results_F/" + suff + save_suff+"/"

   drange = [0, 50]

   d_vx = scipy.io.loadmat(net_data_dirF + "beta_test.mat")
   uF, predF= (d_vx["U_res"].astype(np.float32), d_vx["S_pred"].astype(np.float32))


   err = np.linalg.norm(s[0:400, :].flatten()-predF[0:400, :].flatten(), 2)/np.linalg.norm(s[0:400, :].flatten(), 2)
   RMSE = np.sqrt(np.mean((s[0:400, :].flatten()-predF[0:400, :].flatten())**2))
   
   print(err)
   print(RMSE)
   print("\n")
        
        



        
        

if __name__ == "__main__":
    replay = False
    MAS = False
    RDPS = False 
    scaled = False
 #   run_MF(replay, MAS, RDPS, scaled, N=100) #MF
        
    replay = True
    MAS = False
    RDPS = True 
    scaled = False
   # run_MF(replay, MAS, RDPS, scaled, N=25) #MF-replay
  #  run_MF(replay, MAS, RDPS, scaled, N=50) #MF-replay
  #  run_MF(replay, MAS, RDPS, scaled, N=100) #MF-replay
  #  run_MF(replay, MAS, RDPS, scaled, N=150) #MF-replay
  #  run_MF(replay, MAS, RDPS, scaled, N=200) #MF-replay
        
    replay = False
    MAS = True
    RDPS = True 
    scaled = False
    run_MF(replay, MAS, RDPS, scaled, N=100, MASl=0.001) #MF-MAS
    run_MF(replay, MAS, RDPS, scaled, N=100, MASl=0.01) #MF-MAS
    run_MF(replay, MAS, RDPS, scaled, N=100, MASl=.1) #MF-MAS
    run_MF(replay, MAS, RDPS, scaled, N=100, MASl=1) #MF-MAS
    run_MF(replay, MAS, RDPS, scaled, N=100, MASl=10) #MF-MAS
    run_MF(replay, MAS, RDPS, scaled, N=100, MASl=100) #MF-MAS
        
    replay = False
    MAS = True
    RDPS = True 
    scaled = True
 #   run_MF(replay, MAS, RDPS, scaled, N=100, MASl=0.001) #MF-MAS
 #   run_MF(replay, MAS, RDPS, scaled, N=100, MASl=0.01) #MF-MAS
 #   run_MF(replay, MAS, RDPS, scaled, N=100, MASl=.1) #MF-MAS
 #   run_MF(replay, MAS, RDPS, scaled, N=100, MASl=1) #MF-MAS
 #   run_MF(replay, MAS, RDPS, scaled, N=100, MASl=10) #MF-MAS
 #   run_MF(replay, MAS, RDPS, scaled, N=100, MASl=100) #MF-MAS
        
    replay = False
    MAS = False
    RDPS = True 
    scaled = False
#    run_MF(replay, MAS, RDPS, scaled, N=100) #MF-RDPS