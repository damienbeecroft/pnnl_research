"""
Created on July 2021
@author: Qizhi He (qizhi.he@pnnl.gov)
Note: 
* For multiple beta cases
* For GPU runing
* Add save log_loss during training
* <2021.07.01> Modified the DeepOnet structure based on Xuhui Meng's code ["fs_v2"]
"""

import os

#import numpy as np
import scipy.io
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import time
from utils_fs_v2 import timing
from SF_funcs import DataGenerator, gen_prev_data_B, DataGenerator_ICS, DataGenerator_ICS_A, DNN_class
import math
import jax
import jax.numpy as np
from jax import random, grad, vmap, jit, hessian
from jax.experimental import optimizers
from jax.experimental.ode import odeint
from jax.nn import relu, selu
from jax.config import config
from jax.ops import index_update, index
from jax import lax
from jax.flatten_util import ravel_pytree

import itertools
from functools import partial
from torch.utils import data
from tqdm import trange, tqdm
#import matplotlib.pyplot as plt
from scipy.interpolate import griddata
#import matplotlib
#import math
#import matplotlib.pyplot as plt
#import numpy as np

def save_data(model, params, save_results_to):
    nn = 50
    dom_coords = np.array([[0.0, 0.0],[1.0, 1.0]])
    t = np.linspace(dom_coords[0, 0], dom_coords[1, 0], nn)
    x = np.linspace(dom_coords[0, 1], dom_coords[1, 1], nn)
    t, x = np.meshgrid(t, x)
    X_star = np.hstack((t.flatten()[:, None], x.flatten()[:, None]))
    
    u_star = u(X_star, a,c)
    
    # Predictions
    u_pred = model.predict_u(params, X_star)
    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)

    print('Relative L2 error_u: %e' % (error_u))
    U_pred = griddata(X_star, u_pred.flatten(), (t, x), method='cubic')
    U_star = griddata(X_star, u_star.flatten(), (t, x), method='cubic')

    fname= save_results_to +"beta_test.mat"
    scipy.io.savemat(fname, {'t':t,
                              'x':x, 
                              'U_star':U_star, 
                              'U_pred':U_pred})
    





# Define the exact solution and its derivatives
def u(x, a, c):
    """
    :param x: x = (t, x)
    """
    t = x[:,0:1]
    x = x[:,1:2]
    return np.sin(np.pi * x) * np.cos(c * np.pi * t) + \
            a * np.sin(2 * c * np.pi* x) * np.cos(4 * c  * np.pi * t)

def u_t(x,a, c):
    t = x[:,0:1]
    x = x[:,1:2]
    u_t = -  c * np.pi * np.sin(np.pi * x) * np.sin(c * np.pi * t) - \
            a * 4 * c * np.pi * np.sin(2 * c * np.pi* x) * np.sin(4 * c * np.pi * t)
    return u_t

def u_tt(x, a, c):
    t = x[:,0:1]
    x = x[:,1:2]
    u_tt = -(c * np.pi)**2 * np.sin( np.pi * x) * np.cos(c * np.pi * t) - \
            a * (4 * c * np.pi)**2 *  np.sin(2 * c * np.pi* x) * np.cos(4 * c * np.pi * t)
    return u_tt

def u_xx(x, a, c):
    t = x[:,0:1]
    x = x[:,1:2]
    u_xx = - np.pi**2 * np.sin( np.pi * x) * np.cos(c * np.pi * t) - \
              a * (2 * c * np.pi)** 2 * np.sin(2 * c * np.pi* x) * np.cos(4 * c * np.pi * t)
    return  u_xx


def r(x, a, c):
    return u_tt(x, a, c) - c**2 * u_xx(x, a, c)
                
# =============================================
# =============================================
    
if __name__ == "__main__":
    
    N_low = 50
    layers = [2, N_low, N_low, N_low, N_low, N_low, 1]
    a = 0.5
    c = 2
    batch_size = 30
    batch_size_s = 6
    epochs = 100000
    lr = optimizers.exponential_decay(1e-4, decay_steps=2000, decay_rate=0.99)
    ics_weight = 1
    res_weight = 1
    ut_weight = 1
    EWC_num_samples = 2000


    ymin_A = 0.0
    ymin_B = 0.1
    ymin_C = 0.3
    ymin_D = 0.4
    ymin_E = 0.5
    
    reloadA = False
    reloadB = False
    reloadC = False
    reloadD = False
    reloadE = False
    
    l = 10
    

    # ====================================
    # saving settings
    # ====================================
    save_str = "SF"
    results_dir_A = "../results_A/"+save_str+"/"
    if not os.path.exists(results_dir_A):
        os.makedirs(results_dir_A)
    results_dir_B = "../results_B/"+save_str+"/"
    if not os.path.exists(results_dir_B):
        os.makedirs(results_dir_B)
    results_dir_C= "../results_C/"+save_str+"/"
    if not os.path.exists(results_dir_C):
        os.makedirs(results_dir_C)
    results_dir_D= "../results_D/"+save_str+"/"
    if not os.path.exists(results_dir_D):
        os.makedirs(results_dir_D)            


    # ====================================
    # DNN model A
    # ====================================
    model_A = DNN_class(layers, ics_weight, res_weight, ut_weight, lr, restart =0)
    ics_coords = np.array([[ymin_A, 0.0],[ymin_A, 1.0]])
    bc1_coords = np.array([[ymin_A, 0.0],[ymin_B, 0.0]])
    bc2_coords = np.array([[ymin_A, 1.0],[ymin_B, 1.0]])
    dom_coords = np.array([[ymin_A, 0.0],[ymin_B, 1.0]])

    ics_sampler = DataGenerator_ICS_A(2, ics_coords, lambda x: u(x, a, c), lambda x: u_t(x, a, c), batch_size)
    bc1 = DataGenerator(2, bc1_coords, lambda x: u(x, a, c), batch_size_s)
    bc2 = DataGenerator(2, bc2_coords, lambda x: u(x, a, c), batch_size_s)
    res_sampler = DataGenerator(2, dom_coords, lambda x: r(x, a, c), batch_size)

    if reloadA:
        params_A = model_A.unravel_params(np.load(results_dir_A + '/params.npy'))
        t = model_A.compute_fisher(params_A, dom_coords,  random.PRNGKey(12345), 
                                  num_samples=0, plot_diffs=False, disp_freq=40)
        _, unravel  = ravel_pytree(t)
        d_vx = scipy.io.loadmat(results_dir_A + "/EWC.mat")
        EWC_A  = unravel(d_vx["EWC"][0, :])
    else:     
        model_A.train(ics_sampler, bc1, bc2, res_sampler, nIter=epochs, F = 0, lam = [])

        print('\n ... A Training done ...')
        scipy.io.savemat(results_dir_A +"losses.mat", 
                     {'training_loss':model_A.loss_training_log,
                      'res_loss':model_A.loss_res_log,
                      'ics_loss':model_A.loss_ics_log,
                      'ut_loss':model_A.loss_ut_log})
    
        params_A = model_A.get_params(model_A.opt_state)
        flat_params, _  = ravel_pytree(model_A.get_params(model_A.opt_state))
        np.save(results_dir_A + 'params.npy', flat_params)
    
        save_data(model_A, params_A, results_dir_A)

        EWC_A =model_A.compute_fisher(params_A, dom_coords, random.PRNGKey(12345), 
                                  num_samples=EWC_num_samples, plot_diffs=True, disp_freq=100)
        flat_EWC, _  = ravel_pytree(EWC_A)
        scipy.io.savemat(results_dir_A +"/EWC.mat",  {'EWC':flat_EWC})
        
            

    # ====================================
    # DNN model B
    # ====================================
    model_B = DNN_class(layers, ics_weight, res_weight, ut_weight, lr, restart =1, params_t = params_A, params_i = params_A)
    model = model_B
    results_dir = results_dir_B
    ics_coords = np.array([[ymin_B, 0.0],[ymin_B, 1.0]])
    bc1_coords = np.array([[ymin_B, 0.0],[ymin_C, 0.0]])
    bc2_coords = np.array([[ymin_B, 1.0],[ymin_C, 1.0]])
    dom_coords = np.array([[ymin_B, 0.0],[ymin_C, 1.0]])

    ics_sampler = DataGenerator_ICS(2, ics_coords, model_A, params_A, batch_size)
    bc1 = DataGenerator(2, bc1_coords, lambda x: u(x, a, c), batch_size_s)
    bc2 = DataGenerator(2, bc2_coords, lambda x: u(x, a, c), batch_size_s)
    res_sampler = DataGenerator(2, dom_coords, lambda x: r(x, a, c), batch_size)

    if reloadB:
        params_B = model.unravel_params(np.load(results_dir + '/params.npy'))
        t = model.compute_fisher(params_B, dom_coords,  random.PRNGKey(12345), 
                                  num_samples=0, plot_diffs=False, disp_freq=40)
        _, unravel  = ravel_pytree(t)
        d_vx = scipy.io.loadmat(results_dir + "/EWC.mat")
        EWC_B  = unravel(d_vx["EWC"][0, :])
    else:     
        model.train(ics_sampler, bc1, bc2, res_sampler, nIter=epochs, F = EWC_A, lam = [l])

        print('\n ... B Training done ...')
        scipy.io.savemat(results_dir +"losses.mat", 
                     {'training_loss':model_B.loss_training_log,
                      'res_loss':model.loss_res_log,
                      'ics_loss':model.loss_ics_log,
                      'ut_loss':model.loss_ut_log})
    
        params_B = model.get_params(model.opt_state)
        flat_params, _  = ravel_pytree(params_B)
        np.save(results_dir + 'params.npy', flat_params)
        save_data(model, params_B, results_dir)

        EWC_B =model.compute_fisher(params_B, dom_coords, random.PRNGKey(12345), 
                                  num_samples=EWC_num_samples, plot_diffs=True, disp_freq=100)
        flat_EWC, _  = ravel_pytree(EWC_B)
        scipy.io.savemat(results_dir +"/EWC.mat",  {'EWC':flat_EWC})
        
        
    model_B = model
        
    # ====================================
    # DNN model C
    # ====================================
    EWC = EWC_B + EWC_A
    params_in =  params_B + params_A

    model_C = DNN_class(layers, ics_weight, res_weight, ut_weight, lr, restart =1, params_t = params_in, params_i = params_B)
    model = model_C
    results_dir = results_dir_C
    ymin = ymin_C
    ymax = ymin_D
     
    ics_coords = np.array([[ymin, 0.0],[ymin, 1.0]])
    bc1_coords = np.array([[ymin, 0.0],[ymax, 0.0]])
    bc2_coords = np.array([[ymin, 1.0],[ymax, 1.0]])
    dom_coords = np.array([[ymin, 0.0],[ymax, 1.0]])
    
    ics_sampler = DataGenerator_ICS(2, ics_coords, model_B, params_B, batch_size)
    bc1 = DataGenerator(2, bc1_coords, lambda x: u(x, a, c), batch_size_s)
    bc2 = DataGenerator(2, bc2_coords, lambda x: u(x, a, c), batch_size_s)
    res_sampler = DataGenerator(2, dom_coords, lambda x: r(x, a, c), batch_size)
    
    if reloadC:
         params_C = model.unravel_params(np.load(results_dir + '/params.npy'))
         t = model.compute_fisher(params_C, dom_coords,  random.PRNGKey(12345), 
                                  num_samples=0, plot_diffs=False, disp_freq=40)
         _, unravel  = ravel_pytree(t)
         d_vx = scipy.io.loadmat(results_dir + "/EWC.mat")
         EWC_C  = unravel(d_vx["EWC"][0, :])
    else:     
         model.train(ics_sampler, bc1, bc2, res_sampler, nIter=epochs, F = EWC, lam = [l, l])
    
         print('\n ... C Training done ...')
         scipy.io.savemat(results_dir +"losses.mat", 
                      {'training_loss':model.loss_training_log,
                       'res_loss':model.loss_res_log,
                       'ics_loss':model.loss_ics_log,
                       'ut_loss':model.loss_ut_log})
     
         params_C = model.get_params(model.opt_state)
         flat_params, _  = ravel_pytree(params_C)
         np.save(results_dir + 'params.npy', flat_params)
         save_data(model, params_C, results_dir)
    
         EWC_C =model.compute_fisher(params_C, dom_coords, random.PRNGKey(12345), 
                                  num_samples=EWC_num_samples, plot_diffs=True, disp_freq=100)
         flat_EWC, _  = ravel_pytree(EWC_C)
         scipy.io.savemat(results_dir +"/EWC.mat",  {'EWC':flat_EWC})
    model_C = model 
    
    # ====================================
    # DNN model C
    # ====================================
    EWC = EWC_C + EWC_B + EWC_A
    params_in = params_C + params_B + params_A
    
    model_D = DNN_class(layers, ics_weight, res_weight, ut_weight, lr, restart =1, params_t = params_in, params_i = params_C)
    model = model_D
    results_dir = results_dir_D
    ymin = ymin_D
    ymax = ymin_E
     
    ics_coords = np.array([[ymin, 0.0],[ymin, 1.0]])
    bc1_coords = np.array([[ymin, 0.0],[ymax, 0.0]])
    bc2_coords = np.array([[ymin, 1.0],[ymax, 1.0]])
    dom_coords = np.array([[ymin, 0.0],[ymax, 1.0]])
    
    ics_sampler = DataGenerator_ICS(2, ics_coords, model_C, params_C, batch_size)
    bc1 = DataGenerator(2, bc1_coords, lambda x: u(x, a, c), batch_size_s)
    bc2 = DataGenerator(2, bc2_coords, lambda x: u(x, a, c), batch_size_s)
    res_sampler = DataGenerator(2, dom_coords, lambda x: r(x, a, c), batch_size)
    
    if reloadC:
         params_D = model.unravel_params(np.load(results_dir + '/params.npy'))
         t = model.compute_fisher(params_D, dom_coords,  random.PRNGKey(12345), 
                                  num_samples=0, plot_diffs=False, disp_freq=40)
         _, unravel  = ravel_pytree(t)
         d_vx = scipy.io.loadmat(results_dir + "/EWC.mat")
         EWC_D  = unravel(d_vx["EWC"][0, :])
    else:     
         model.train(ics_sampler, bc1, bc2, res_sampler, nIter=epochs, F = EWC, lam = [l, l, l])
    
         print('\n ... D Training done ...')
         scipy.io.savemat(results_dir +"losses.mat", 
                      {'training_loss':model.loss_training_log,
                       'res_loss':model.loss_res_log,
                       'ics_loss':model.loss_ics_log,
                       'ut_loss':model.loss_ut_log})
     
         params_D = model.get_params(model.opt_state)
         flat_params, _  = ravel_pytree(params_D)
         np.save(results_dir + 'params.npy', flat_params)
         save_data(model, params_D, results_dir)
    
         EWC_D =model.compute_fisher(params_D, dom_coords, random.PRNGKey(12345), 
                                  num_samples=EWC_num_samples, plot_diffs=True, disp_freq=100)
         flat_EWC, _  = ravel_pytree(EWC_D)
         scipy.io.savemat(results_dir +"/EWC.mat",  {'EWC':flat_EWC})
         
                