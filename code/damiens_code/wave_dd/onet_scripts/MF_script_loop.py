
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
from jax.example_libraries import optimizers
from jax.nn import relu, selu
from jax.config import config
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

def save_data_MF(model, params, save_results_to):
    nn = 100
    dom_coords = np.array([[0.0, 0.0],[1.0, 1.0]])
    t = np.linspace(dom_coords[0, 0], dom_coords[1, 0], nn)
    x = np.linspace(dom_coords[0, 1], dom_coords[1, 1], nn)
    t, x = np.meshgrid(t, x)
    X_star = np.hstack((t.flatten()[:, None], x.flatten()[:, None]))
    
    u_star = u(X_star, a,c)
    # Predictions
    u_pred = model.predict_u(params, X_star)

    U_pred = griddata(X_star, u_pred.flatten(), (t, x), method='cubic')
    U_star = griddata(X_star, u_star.flatten(), (t, x), method='cubic')
    error_u = np.linalg.norm(U_star - U_pred, 2) / np.linalg.norm(U_star, 2)
    print('Relative L2 error_u: %e' % (error_u))

    fname= save_results_to +"beta_test.mat"
    scipy.io.savemat(fname, {'t':t,
                              'x':x, 
                              'U_star':U_star, 
                              'U_pred':U_pred}, format='4')

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

    U_pred = griddata(X_star, u_pred.flatten(), (t, x), method='cubic')
    U_star = griddata(X_star, u_star.flatten(), (t, x), method='cubic')
    error_u = np.linalg.norm(U_star - U_pred, 2) / np.linalg.norm(U_star, 2)
    print('Relative L2 error_u: %e' % (error_u))
    
    fname= save_results_to +"beta_test.mat"
    scipy.io.savemat(fname, {'t':t,
                              'x':x, 
                              'U_star':U_star, 
                              'U_pred':U_pred}, format='4')
    

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
    def __init__(self, dim, coords, res_pts, err_norm, func,
                 batch_size=64, batch_size_res=64, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.dim = dim
        self.coords = coords
        self.func = func
        
        self.res_pts = res_pts
        self.N = res_pts.shape[0]
        self.res_val = res_val
        self.batch_size_res = batch_size_res

        
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
        
        idx = random.choice(key, self.N, (self.batch_size_res,), p=self.res_val, replace=False)
        x_res = self.res_pts[idx]
        x = np.concatenate([x, x_res])
        
        y = self.func(x)
        inputs = (x)
        outputs = y
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
        inputs = (x)
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
    N_low= 200
    layer_sizes_nl = [3, N_low, N_low, N_low, 1]
    layer_sizes_l = [1, 1]
    
    a = 0.5
    c = 2
    # batch_size = 300
    batch_size = 1000
    batch_size_s = 300
    epochs = 100000
    # epochsA2 = 10
    epochsA2 = 500000
    lr = optimizers.exponential_decay(5e-4, decay_steps=2000, decay_rate=0.99)
    lrA = optimizers.exponential_decay(1e-4, decay_steps=2000, decay_rate=0.99)
    ics_weight = 1
    res_weight = 10.0
    ut_weight = 1


    ymin_A = 0.0
    ymin_B = 1.0

    #==== parameters that I am adding =====
    delta = 1.9
    #======================================

    steps_to_train = np.arange(1)
    reload = [False]
    
    reloadA = True

    
    l = 0
    

    # ====================================
    # saving settings
    # ====================================
    # path_to_wave = "C:/Users/beec613/Desktop/pnnl_research/code/damiens_code/wave_dd"
    path_to_wave = "/people/beec613/pnnl_research/code/damiens_code/wave_dd"
    save_str = "MF_loop"
    results_dir_A = path_to_wave + "/results_A/" + save_str
    if not os.path.exists(results_dir_A):
        os.makedirs(results_dir_A)

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
    params_prev = []
    
#    ics_sampler = DataGenerator_ICS_MFA2(2, ics_coords,  lambda x: u(x, a, c), lambda x: u_t(x, a, c), batch_size)

#    bc1 = DataGenerator_MF(2, bc1_coords, lambda x: u(x, a, c), batch_size_s)
#    bc2 = DataGenerator_MF(2, bc2_coords, lambda x: u(x, a, c), batch_size_s)
    k = 2
    c = 0 
    key = random.PRNGKey(1234)
    batch_size_res = int(batch_size/2)    
    batch_size_pts = batch_size - batch_size_res
    
    key, subkey = random.split(key)
    res_pts = dom_coords[0:1,:] + (dom_coords[1:2,:]-dom_coords[0:1,:])*random.uniform(key, shape=[20000, 2])
    res_val = model_A.predict_res(params_A, res_pts)
    err = res_val**k/np.mean( res_val**k) + c
    err_norm = err/np.sum(err)                        
    res_sampler = DataGenerator_MF(2, dom_coords, res_pts, err_norm, lambda x: r(x, a, c), batch_size_pts, batch_size_res)
    
    Ndomains = []
    for step in steps_to_train:
        results_dir = path_to_wave + "/results_" + str(step) + "/"+save_str+"/"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        if step == 0:
            res = 0
        else:
            res = 1

        Ndomains.append(4**(step+1))
        model = MF_DNN_class(layer_sizes_nl, layer_sizes_l,layers, ics_weight, res_weight, 
                             ut_weight, lr, Ndomains, delta, dom_coords, params_A, 
                             params_t=params_prev, restart=res)
        
        if reload[step]:
            params = model.unravel_params(np.load(results_dir + '/params.npy'))
        
        else:     
            model.train(ics_sampler, bc1, bc2, res_sampler, nIter=epochsA2, F = 0, lam = [])
        
            print('\n ... A2 Training done ...')
            scipy.io.savemat(results_dir +"losses.mat", 
                         {'training_loss':model.loss_training_log,
                          'res_loss':model.loss_res_log,
                          'ics_loss':model.loss_ics_log,
                          'ut_loss':model.loss_ut_log})
        
            params = model.get_params(model.opt_state)
            flat_params, _  = ravel_pytree(params)
            np.save(results_dir + 'params.npy', flat_params)
        
            save_data_MF(model, params, results_dir)
            
        params_prev.append(params)
        
        key, subkey = random.split(key)
        res_pts = dom_coords[0:1,:] + (dom_coords[1:2,:]-dom_coords[0:1,:])*random.uniform(key, shape=[20000, 2])
        res_val = model.predict_res(params, res_pts)
        err = res_val**k/np.mean( res_val**k) + c
        err_norm = err/np.sum(err)                        
        res_sampler = DataGenerator_MF(2, dom_coords, res_pts, err_norm, lambda x: r(x, a, c), batch_size_pts, batch_size_res)
      
        
        
        
        
        
        
        
   
                
