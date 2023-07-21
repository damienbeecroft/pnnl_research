#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 16:24:15 2019

@author: howa549
"""

import os

import numpy as onp
import scipy.io
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import jax
# import jax.numpy as np
import time
from utils_fs_v2 import timing,  DataGenerator, DataGenerator_res, DataGenerator_res2
import math
import jax
import jax.numpy as np
from jax import random, grad, vmap, jit, hessian
from jax.example_libraries import optimizers
from jax.experimental.ode import odeint
from jax.nn import relu, selu
from jax.config import config
#from jax.ops import index_update, index
from jax import lax
from jax.flatten_util import ravel_pytree

import itertools
from functools import partial
from torch.utils import data
from tqdm import trange, tqdm
#import matplotlib.pyplot as plt
#import pandas as pd

from DNN_EWC_Class import DNN_class_EWC
from MF_EWC_Class import MF_class_EWC


def save_data(model, params,  save_results_to, save_prfx):
    # ====================================
    # Saving model
    # ====================================
    t_train_range = np.linspace(0, 50, 2000)
    u_res = t_train_range.reshape([len(t_train_range), 1])
    flat_params, _  = ravel_pytree(model.get_params(model.opt_state))
    np.save(save_results_to + 'params_' + save_prfx + '.npy', flat_params)

    S_pred =  model.predict_full(params, u_res)

    fname= save_results_to +"beta_test.mat"
    scipy.io.savemat(fname, {'U_res':u_res,'S_pred':S_pred})
    
    scipy.io.savemat(save_results_to +"losses.mat", 
                     {'training_loss':model.loss_training_log,
                      'res_loss':model.loss_res_log,
                      'ics_loss':model.loss_ics_log,
                      'data_loss':model.loss_data_log})

def save_dataDNN(model, params,  save_results_to, save_prfx):
    # ===================================
    # Saving model
    # ====================================
    t_train_range = np.linspace(0, 50, 2000)
    u_res = t_train_range.reshape([len(t_train_range), 1])
    flat_params, _  = ravel_pytree(model.get_params(model.opt_state))
    np.save(save_results_to + 'params_' + save_prfx + '.npy', flat_params)

    S_pred =  model.predict_low(params, u_res)

    fname= save_results_to +"beta_test.mat"
    scipy.io.savemat(fname, {'U_res':u_res,'S_pred':S_pred})
    
    scipy.io.savemat(save_results_to +"losses.mat", 
                     {'training_loss':model.loss_training_log,
                      'res_loss':model.loss_res_log,
                      'ics_loss':model.loss_ics_log,
                      'data_loss':model.loss_data_log})

    

if __name__ == "__main__":
    ics_weight = 1.0
    res_weight = 1.0 
    data_weight  = 0.0
    pen_weight  = 0.000001

    batch_size = 100
    batch_size_res = int(batch_size/2)

    steps_to_train = np.arange(6)
    reload = [True, True, True, False, False, False]
    
    reloadA = True
    
    

    k = 2
    c = 0 


    epochs = 1000
    epochsA2 = 100000
    lr = optimizers.exponential_decay(1e-3, decay_steps=2000, decay_rate=0.99)
    N_low = 200 
    N_nl = 80
    layers_A = [1, N_low, N_low, N_low, 2]
    layers_sizes_nl = [3, N_nl, N_nl, N_nl, 2]
    layers_sizes_l = [2,  4, 2]

    min_A = 0
    min_B = 10
    Tmax = min_B
    delta = 1.9

    data_range = np.arange(0,int(2*min_B))


    # d_vx = scipy.io.loadmat("../data.mat") # This was the original line
    d_vx = scipy.io.loadmat("C:/Users/beec613/Desktop/pnnl_research/code/amandas_code/Pendulum_DD/data.mat")
    t_data_full, s_data_full = (d_vx["u"].astype(np.float32), 
               d_vx["s"].astype(np.float32))

    # ====================================
    # saving settings
    # ====================================
    save_str = "MF_loop"
    results_dir_A = "../results_A/"+save_str+"/"
    if not os.path.exists(results_dir_A):
        os.makedirs(results_dir_A)
        
    # ====================================
    # Train A
    # ====================================
    
    u_bc = np.asarray([0]).reshape([1, -1])
    s_bc = np.asarray([1, 1]).reshape([1, -1])
    u_bc = jax.device_put(u_bc)
    s_bc = jax.device_put(s_bc)

    t_data = jax.device_put(t_data_full[:, data_range].reshape([-1, 1]))
    s_data = jax.device_put(s_data_full[data_range, :].reshape([-1, 2]))

    # Create data set
    coords = [min_A, min_B]
    ic_dataset = DataGenerator(u_bc, s_bc, 1)
    res_dataset = DataGenerator_res(coords, batch_size)
    data_dataset = DataGenerator(t_data, s_data, len(t_data))

    lam= []
    F = []
    results_dir = results_dir_A
    model_A = DNN_class_EWC(layers_A, ics_weight, res_weight, data_weight, [], lr)
    if reloadA:
        params_A = model_A.unravel_params(np.load(results_dir+ '/params.npy'))

    
    else:
        model_A.train(ic_dataset, res_dataset, data_dataset, lam, F, [], nIter=epochs)
        params_A = model_A.get_params(model_A.opt_state)
        save_dataDNN(model_A, params_A,  results_dir +"/", 'A')   
        
        flat_params, _  = ravel_pytree(model_A.get_params(model_A.opt_state))
        np.save(results_dir + 'params.npy', flat_params)
        print('\n ... A Training done ...')
        
        scipy.io.savemat(results_dir +"losses.mat", 
                     {'training_loss':model_A.loss_training_log,
                      'res_loss':model_A.loss_res_log,
                      'ics_loss':model_A.loss_ics_log,
                      'ut_loss':model_A.loss_data_log}, format='4')
    

    # ====================================
    # DNN model A2
    # ====================================
  #  res_weight = 100.0
    params_prev = []
    

    k = 2
    c = 0 
    key = random.PRNGKey(1234)
    batch_size_res = int(batch_size/2)    
    batch_size_pts = batch_size - batch_size_res                            
                                     
    
    key, subkey = random.split(key)

    res_pts = coords[0] + (coords[1]-coords[0])*random.uniform(key, shape=[20000,1])
    res_val = model_A.predict_res(params_A, res_pts)
    err = res_val**k/np.mean( res_val**k) + c
    err_norm = err/np.sum(err)                        
    res_dataset = DataGenerator_res2(coords, res_pts, err_norm, batch_size_res, batch_size)
    
    
    Ndomains = []
    for step in steps_to_train:
        # results_dir = "../results_" + str(step) + "/"+save_str+"/" # This is the original line
        results_dir = "C:/Users/beec613/Desktop/pnnl_research/code/amandas_code/Pendulum_DD/results_" + str(step) + "/"+save_str+"/"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        res = 0
        if step > 0:
            res=1
            
        Ndomains.append(2**(step+1))

        sigma = Tmax*delta/(2*(Ndomains[-1] - 1))
        mus = Tmax*onp.linspace(0,1,Ndomains[-1])
        double_domains = onp.array([[mus[j+1] - sigma, mus[j] + sigma] for j in range(Ndomains[-1] - 1)])
        single_domains = onp.array([[mus[j] + sigma, mus[j+2] - sigma] for j in range(Ndomains[-1] - 2)])
 
        model = MF_class_EWC(layers_sizes_nl, layers_sizes_l, layers_A, ics_weight, 
                         res_weight, data_weight, pen_weight,lr, Ndomains, delta, Tmax, params_A, params_t = params_prev, restart =res)

        
        if reload[step]:
            params = model.unravel_params(np.load(results_dir + '/params.npy'))

        
        else:     
            model.train(ic_dataset, res_dataset, data_dataset, nIter=epochsA2)
        



            print('\n ... Level ' + str(step) + ' Training done ...')
            scipy.io.savemat(results_dir +"losses.mat", 
                         {'training_loss':model.loss_training_log,
                          'res_loss':model.loss_res_log,
                          'ics_loss':model.loss_ics_log,
                          'ut_loss':model.loss_data_log})
        
            params = model.get_params(model.opt_state)
            flat_params, _  = ravel_pytree(params)
            np.save(results_dir + 'params.npy', flat_params)
        
            save_data(model,  params, results_dir, 'B')   
            
        params_prev.append(params)
        
        key, subkey = random.split(key)
        res_pts = coords[0] + (coords[1]-coords[0])*random.uniform(key, shape=[20000,1])
        res_val = model.predict_res(params, res_pts)
        err = res_val**k/np.mean( res_val**k) + c
        err_norm = err/np.sum(err)                        
      
        res_dataset = DataGenerator_res2(coords, res_pts, err_norm, batch_size_res, batch_size)

      
            




    
# =============================================
# =============================================
#if __name__ == "__main__":
    
    replay = False
    MAS = False
    RDPS = False 
    scaled = False
   # run_MF(replay, MAS, RDPS, scaled) #MF
    