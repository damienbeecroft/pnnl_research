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
import scipy.io

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
    y = np.linspace(0, 50, 1001)
    s = odeint(system, u.flatten(), y)
    skip = np.arange(0, 1000, 10)

    
    fig3, gx = plt.subplots(figsize=(6, 4))
    plt.figure(fig3.number)
    plt.plot(y, s[:, 0], '#59a14f', linestyle='-', linewidth=1)
    plt.plot(y, s[:, 1], '#4e79a7', linestyle='-', linewidth=1)
    
    
    fname= "data.mat"
    scipy.io.savemat(fname, {'u':y[skip],
                              's':s[skip, :]})
       