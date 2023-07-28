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
from MF_funcs import DNN_class, MF_DNN_class
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

def save_data_MF(model, params, save_results_to, func=0):
    nn = 100
    dom_coords = np.array([[0.0, 0.0],[1.0, 1.0]])
    t = np.linspace(dom_coords[0, 0], dom_coords[1, 0], nn)
    x = np.linspace(dom_coords[0, 1], dom_coords[1, 1], nn)
    t, x = np.meshgrid(t, x)
    X_star = np.hstack((t.flatten()[:, None], x.flatten()[:, None]))
    
    u_star = u(X_star, a,c)
    yprev = func(X_star)
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

def save_data(model, params, save_results_to):
    nn = 100
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
    

    



def gen_prev_data_A2(u, model_A, params_A):
    predA = model_A.predict_u(params_A, u)
    return predA

def gen_prev_data_A2_ut(u, model_A, params_A):
    predAt = model_A.predict_ut(params_A, u)
    return predAt


def gen_prev_data_B(u,  model_B, params_B):
    predB = model_B.predict_u(params_B, u)
    return predB

def gen_prev_data_B_ut(u, model_B, params_B):
    predB = model_B.predict_ut(params_B, u)
    return predB




class DataGenerator(data.Dataset):
    def __init__(self, dim, coords, func, 
                 batch_size=64, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.dim = dim
        self.coords = coords
        self.func = func
        
        self.batch_size = batch_size
        self.key = rng_key

    def __getitem__(self, index):
        'Generate one batch of data'
        self.key, subkey = random.split(self.key)
        inputs, outputs = self.__data_generation(subkey)
        return inputs, outputs

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key):
        'Generates data containing batch_size samples'
        x = self.coords[0:1,:] + (self.coords[1:2,:]-self.coords[0:1,:])*random.uniform(key, shape=[self.batch_size, self.dim])
        y = self.func(x)
        inputs = x
        outputs = y
        return inputs, outputs
    


class DataGenerator_ICS(data.Dataset):
    def __init__(self, dim, coords, func, func_ut,
                 batch_size=64, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.dim = dim
        self.coords = coords
        self.func = func
        self.func_ut = func_ut
        
        self.batch_size = batch_size
        self.key = rng_key

    def __getitem__(self, index):
        'Generate one batch of data'
        self.key, subkey = random.split(self.key)
        inputs, outputs = self.__data_generation(subkey)
        return inputs, outputs

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key):
        'Generates data containing batch_size samples'
        x = self.coords[0:1,:] + (self.coords[1:2,:]-self.coords[0:1,:])*random.uniform(key, shape=[self.batch_size, self.dim])
        y = self.func(x)
        yt = self.func_ut(x)
        inputs = x
        outputs = (y, yt)
        return inputs, outputs

class DataGenerator_MF(data.Dataset):
    def __init__(self, dim, coords, func,
                 batch_size=64, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.dim = dim
        self.coords = coords
        self.func = func
        
        self.batch_size = batch_size
        self.key = rng_key

    def __getitem__(self, index):
        'Generate one batch of data'
        self.key, subkey = random.split(self.key)
        inputs, outputs = self.__data_generation(subkey)
        return inputs, outputs

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key):
        'Generates data containing batch_size samples'
        x = self.coords[0:1,:] + (self.coords[1:2,:]-self.coords[0:1,:])*random.uniform(key, shape=[self.batch_size, self.dim])
        y = self.func(x)
        yprev = 0.0
        inputs = (x, yprev)
        outputs = y
        return inputs, outputs
    


class DataGenerator_ICS_MF(data.Dataset):
    def __init__(self, dim, coords, func, func_ut,
                 batch_size=64, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.dim = dim
        self.coords = coords
        self.func = func
        self.func_ut = func_ut
        
        self.batch_size = batch_size
        self.key = rng_key

    def __getitem__(self, index):
        'Generate one batch of data'
        self.key, subkey = random.split(self.key)
        inputs, outputs = self.__data_generation(subkey)
        return inputs, outputs

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key):
        'Generates data containing batch_size samples'
        x = self.coords[0:1,:] + (self.coords[1:2,:]-self.coords[0:1,:])*random.uniform(key, shape=[self.batch_size, self.dim])
        y = self.func(x)
        yt = self.func_ut(x)
        inputs = (x, y)
        outputs = (y, yt)
        return inputs, outputs

    
class DataGenerator_ICS_MFA2(data.Dataset):
    def __init__(self, dim, coords, func, func_ut,
                 batch_size=64, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.dim = dim
        self.coords = coords
        self.func = func
        self.func_ut = func_ut
        
        self.batch_size = batch_size
        self.key = rng_key

    def __getitem__(self, index):
        'Generate one batch of data'
        self.key, subkey = random.split(self.key)
        inputs, outputs = self.__data_generation(subkey)
        return inputs, outputs

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key):
        'Generates data containing batch_size samples'
        x = self.coords[0:1,:] + (self.coords[1:2,:]-self.coords[0:1,:])*random.uniform(key, shape=[self.batch_size, self.dim])
        yprev = 0.0
        y = self.func(x)
        yt = self.func_ut(x)
        inputs = (x, yprev)
        outputs = (y, yt)
        return inputs, outputs


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
    
    N_low = 500
    layers = [2, N_low, N_low, N_low, N_low, N_low, 1]
    N_low=500
    layer_sizes_nl = [3, N_low, N_low, N_low, 1]
    layer_sizes_l = [1, 1]
    
    a = 0.5
    c = 2
    batch_size = 300
    batch_size_s = 300
    epochs = 100000
    epochsA2 = 200000
    lr = optimizers.exponential_decay(5e-4, decay_steps=2000, decay_rate=0.99)
    lrA = optimizers.exponential_decay(1e-4, decay_steps=2000, decay_rate=0.99)
    ics_weight = 1
    res_weight = 1.0
    ut_weight = 1
    EWC_num_samples = 10


    ymin_A = 0.0
    ymin_B = 1.0
    ymin_C = 1.0
    ymin_D = 0.3
    ymin_E = 0.4
    
    reloadA = False
    reloadA2 = False
    reloadB = False
    reloadC = False
    reloadD = True
    reloadE = True
    
    l = 0
    

    # ====================================
    # saving settings
    # ====================================
    save_str = "MF_more"
    results_dir_A = "../results_A/"+save_str+"/"
    if not os.path.exists(results_dir_A):
        os.makedirs(results_dir_A)
    results_dir_A2 = "../results_A2/"+save_str+"/"
    if not os.path.exists(results_dir_A2):
        os.makedirs(results_dir_A2)
    results_dir_B = "../results_B/"+save_str+"/"
    if not os.path.exists(results_dir_B):
        os.makedirs(results_dir_B)
    results_dir_C = "../results_C/"+save_str+"/"
    if not os.path.exists(results_dir_C):
        os.makedirs(results_dir_C)
    results_dir_D = "../results_D/"+save_str+"/"
    if not os.path.exists(results_dir_D):
        os.makedirs(results_dir_D)
                  


    # ====================================
    # DNN model A
    # ====================================

    model_A = DNN_class(layers, ics_weight, res_weight, ut_weight, lrA, restart =0)

    ics_coords = np.array([[ymin_A, 0.0],[ymin_A, 1.0]])
    bc1_coords = np.array([[ymin_A, 0.0],[ymin_B, 0.0]])
    bc2_coords = np.array([[ymin_A, 1.0],[ymin_B, 1.0]])
    dom_coords = np.array([[ymin_A, 0.0],[ymin_B, 1.0]])

    ics_sampler = DataGenerator_ICS(2, ics_coords, lambda x: u(x, a, c), lambda x: u_t(x, a, c), batch_size)
    bc1 = DataGenerator(2, bc1_coords, lambda x: u(x, a, c), batch_size_s)
    bc2 = DataGenerator(2, bc2_coords, lambda x: u(x, a, c), batch_size_s)
    res_sampler = DataGenerator(2, dom_coords, lambda x: r(x, a, c), batch_size)

    if reloadA:
        params_A = model_A.unravel_params(np.load(results_dir_A + '/params.npy'))
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

    # ====================================
    # DNN model A2
    # ====================================
  #  res_weight = 100.0

    model_A2 = MF_DNN_class(layer_sizes_nl, layer_sizes_l,layers, ics_weight, res_weight, ut_weight, lr, params_A, params_t=[], restart =0)
    model = model_A2
    results_dir = results_dir_A2

    ics_sampler = DataGenerator_ICS_MFA2(2, ics_coords,  lambda x: u(x, a, c), lambda x: u_t(x, a, c), 
                                        batch_size)
    
    
    bc1 = DataGenerator_MF(2, bc1_coords, lambda x: u(x, a, c), batch_size_s)
    bc2 = DataGenerator_MF(2, bc2_coords, lambda x: u(x, a, c), batch_size_s)
    res_sampler = DataGenerator_MF(2, dom_coords, lambda x: r(x, a, c), batch_size)
    
    if reloadA2:
        params_A2 = model.unravel_params(np.load(results_dir + '/params.npy'))

    else:     
        model.train(ics_sampler, bc1, bc2, res_sampler, nIter=epochsA2, F = 0, lam = [])

        print('\n ... A2 Training done ...')
        scipy.io.savemat(results_dir +"losses.mat", 
                     {'training_loss':model.loss_training_log,
                      'res_loss':model.loss_res_log,
                      'ics_loss':model.loss_ics_log,
                      'ut_loss':model.loss_ut_log})
    
        params_A2 = model.get_params(model.opt_state)
        flat_params, _  = ravel_pytree(params_A2)
        np.save(results_dir + 'params.npy', flat_params)
    
        save_data_MF(model, params_A2, results_dir, func=lambda x: gen_prev_data_A2(x, model_A, params_A))



    #model_B

    model_B = MF_DNN_class(layer_sizes_nl, layer_sizes_l,layers, ics_weight, res_weight, ut_weight, 
                           lr, params_A, params_t = params_A2, restart =1)
    model = model_B
    results_dir = results_dir_B


    if reloadB:
        params_B = model.unravel_params(np.load(results_dir + '/params.npy'))

    else:     
        model.train(ics_sampler, bc1, bc2, res_sampler, nIter=epochsA2, F = 0, lam = [])

        print('\n ... B Training done ...')
        scipy.io.savemat(results_dir +"losses.mat", 
                     {'training_loss':model.loss_training_log,
                      'res_loss':model.loss_res_log,
                      'ics_loss':model.loss_ics_log,
                      'ut_loss':model.loss_ut_log})
    
        params_B = model.get_params(model.opt_state)
        flat_params, _  = ravel_pytree(params_B)
        np.save(results_dir + 'params.npy', flat_params)
    
        save_data_MF(model, params_B, results_dir, func=lambda x: gen_prev_data_A2(x, model_A2, params_A2))
        
        
    
    #model_C


    params_prev = params_B + params_A2
    
    model_C = MF_DNN_class(layer_sizes_nl, layer_sizes_l,layers, ics_weight, res_weight, ut_weight, 
                           lr, params_A, params_t = params_prev, restart =1)
    model = model_C
    results_dir = results_dir_C


    if reloadC:
        params_C = model.unravel_params(np.load(results_dir + '/params.npy'))

    else:     
        model.train(ics_sampler, bc1, bc2, res_sampler, nIter=epochsA2, F = 0, lam = [])

        print('\n ... C Training done ...')
        scipy.io.savemat(results_dir +"losses.mat", 
                     {'training_loss':model.loss_training_log,
                      'res_loss':model.loss_res_log,
                      'ics_loss':model.loss_ics_log,
                      'ut_loss':model.loss_ut_log})
    
        params_C = model.get_params(model.opt_state)
        flat_params, _  = ravel_pytree(params_C)
        np.save(results_dir + 'params.npy', flat_params)
    
        save_data_MF(model, params_C, results_dir, func=lambda x: gen_prev_data_B(x, model_B, params_B))

    


    #Train D
    params_prev = params_C + params_B + params_A2

    model_D = MF_DNN_class(layer_sizes_nl, layer_sizes_l,layers, ics_weight, res_weight, ut_weight, 
                           lr, params_A, params_t = params_prev, restart =1)
    model = model_D
    results_dir = results_dir_D


    if reloadB:
        params_D = model.unravel_params(np.load(results_dir + '/params.npy'))

    else:     
        model.train(ics_sampler, bc1, bc2, res_sampler, nIter=epochsA2, F = 0, lam = [])

        print('\n ... D Training done ...')
        scipy.io.savemat(results_dir +"losses.mat", 
                     {'training_loss':model.loss_training_log,
                      'res_loss':model.loss_res_log,
                      'ics_loss':model.loss_ics_log,
                      'ut_loss':model.loss_ut_log})
    
        params_D = model.get_params(model.opt_state)
        flat_params, _  = ravel_pytree(params_D)
        np.save(results_dir + 'params.npy', flat_params)
    
        save_data_MF(model, params_D, results_dir, func=lambda x: gen_prev_data_B(x, model_B, params_B))

    



    
                 
                