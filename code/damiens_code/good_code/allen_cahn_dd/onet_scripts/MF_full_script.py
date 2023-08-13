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
import sys
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".XX"
#os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"



#import numpy as np
import scipy.io
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


import time
from utils_fs_v2 import timing
import MF_MAS_H_funcs
from MF_MAS_H_funcs import DNN_class, MF_DNN_class

import math
import jax
import jax.numpy as np
from jax import random, grad, vmap, jit, hessian
from jax.example_libraries import optimizers
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
from scipy.interpolate import griddata
#import matplotlib
#import math
#import matplotlib.pyplot as plt
#import numpy as np
import skopt


def get_pts(model, params, dom_coords, func=0, key=random.PRNGKey(1234), N = 10000):
    key, subkey = random.split(key)
    k = 2
    c = 0
    
    X_star = dom_coords[0:1,:] + (dom_coords[1:2,:]-dom_coords[0:1,:])*\
        random.uniform(key, shape=[N, 2])

    # Predictions
    u_pred_res = model.predict_res(params, X_star)
    err = u_pred_res**k/np.mean( u_pred_res**k) + c
    err_norm = err/np.sum(err)

    return X_star, err_norm

def save_data_MF(model, params, save_results_to, path_to_AC, func=0):
    d_vx = scipy.io.loadmat(path_to_AC + "AC.mat")
    t, x, U_star = (d_vx["tt"].astype(np.float32), 
               d_vx["x"].astype(np.float32), 
               d_vx["uu"].astype(np.float32))
    t, x = np.meshgrid(t[0, :], x[0, :])
    X_star = np.hstack((t.flatten()[:, None], x.flatten()[:, None]))
        
    # Predictions
    u_pred = model.predict_u(params, X_star)
#    u_pred_res = model.predict_res(params, X_star)
    U_pred = griddata(X_star, u_pred.flatten(), (t, x), method='cubic')
#    U_pred_res = griddata(X_star, u_pred_res.flatten(), (t, x), method='cubic')

    fname= save_results_to +"beta_test.mat"
    scipy.io.savemat(fname, {'t':t,
                              'x':x, 
                              'U_star':U_star, 
                              'U_pred':U_pred}, format='4') 
                              #'U_pred':U_pred})
    

def save_data(model, params, save_results_to, path_to_AC):
        d_vx = scipy.io.loadmat(path_to_AC + "AC.mat")
        t, x, U_star = (d_vx["tt"].astype(np.float32), 
                   d_vx["x"].astype(np.float32), 
                   d_vx["uu"].astype(np.float32))
        t, x = np.meshgrid(t[0, :], x[0, :])
        X_star = np.hstack((t.flatten()[:, None], x.flatten()[:, None]))
        
        # Predictions
        u_pred = model.predict_u(params, X_star)
        u_pred_res = model.predict_res(params, X_star)

        U_pred = griddata(X_star, u_pred.flatten(), (t, x), method='cubic')
        U_pred_res = griddata(X_star, u_pred_res.flatten(), (t, x), method='cubic')

        fname= save_results_to +"beta_test.mat"
        scipy.io.savemat(fname, {'t':t,
                                  'x':x, 
                                  'U_star':U_star, 
                                  'U_pred_res':U_pred_res, 
                                  'U_pred':U_pred}, format='4')






class DataGenerator_ICS_A(data.Dataset):
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
    
class DataGenerator_res_A(data.Dataset):
    def __init__(self, dim, coords,
                 batch_size=64, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.dim = dim
        self.coords = coords
        
        self.batch_size = batch_size
        self.key = rng_key
        sampler = skopt.sampler.Hammersly(min_skip=-1, max_skip=-1)
        self.xvals = np.asarray(sampler.generate(
         [(float(self.coords[0, 0]), float(self.coords[1,0])), 
          (float(self.coords[0, 1]), float(self.coords[1, 1]))], 10000) )

    def __getitem__(self, index):
        'Generate one batch of data'
        self.key, subkey = random.split(self.key)
        inputs = self.__data_generation(subkey)
        return inputs

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key):
        'Generates data containing batch_size samples'

        #x = self.coords[0:1,:] + (self.coords[1:2,:]-self.coords[0:1,:])*random.uniform(key, shape=[self.batch_size, self.dim])
        ind = random.choice(key, 10000, (self.batch_size,), replace=False)
        x = self.xvals[ind]
        inputs = x
        return inputs

class DataGenerator_energy(data.Dataset):
    def __init__(self, dim, coords, e_0, N,
                 batch_size=64, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.dim = dim
        self.coords = coords
        self.N  = N
        self.dx = coords[1, 0]/N
        

        self.batch_size = batch_size
        self.key = rng_key
        
        x = np.linspace(self.coords[0, 0], self.coords[1, 0], self.N)
        x = np.reshape(x, [-1, self.N])
        x = np.repeat(x, self.batch_size, axis=0).reshape(self.batch_size, self.N, -1)
        self.x = x
        self.e_0 = e_0*np.ones(self.batch_size)

        

    def __getitem__(self, index):
        'Generate one batch of data'
        self.key, subkey = random.split(self.key)
        inputs, outputs = self.__data_generation(subkey)
        return inputs, outputs

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key):
        'Generates data containing batch_size samples'
        t = self.coords[0,1] + (self.coords[1,1]-self.coords[0,1])*random.uniform(key, shape=[self.batch_size, 1])
        t = np.repeat(t, self.N, axis=1).reshape(self.batch_size, self.N, -1)
        
        
        
        xvec = np.concatenate([t, self.x], axis=2)
        inputs = (xvec, self.dx)
        outputs = self.e_0
        return inputs, outputs



class DataGenerator_BCS_A(data.Dataset):
    def __init__(self, dim, coords1, coords2,
                 batch_size=64, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.dim = dim
        self.coords1 = coords1
        self.coords2 = coords2
        
        self.batch_size = batch_size
        self.key = rng_key

    def __getitem__(self, index):
        'Generate one batch of data'
        self.key, subkey = random.split(self.key)
        inputs = self.__data_generation(subkey)
        return inputs

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key):
        'Generates data containing batch_size samples'
        xval = random.uniform(key, shape=[self.batch_size, self.dim])
        x1 = self.coords1[0:1,:] + (self.coords1[1:2,:]-self.coords1[0:1,:])*xval
        x2 = self.coords2[0:1,:] + (self.coords2[1:2,:]-self.coords2[0:1,:])*xval
        inputs = x1, x2
        return inputs    
    

class DataGenerator_ICS_MF(data.Dataset):
    def __init__(self, dim, coords, func, func_prev,
                 batch_size=64, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.dim = dim
        self.coords = coords
        self.func = func
        self.func_prev = func_prev

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
        outputs = y
        return x, outputs
    
class DataGenerator_res_MF(data.Dataset):
    def __init__(self, dim, coords,
                 batch_size=64, batch_size_res=32, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.dim = dim
        self.coords = coords
        
        self.batch_size = batch_size
        self.batch_size_res = batch_size_res
        self.key = rng_key
        sampler = skopt.sampler.Hammersly(min_skip=-1, max_skip=-1)

        self.xvals = np.asarray(sampler.generate(
         [(float(self.coords[0, 0]), float(self.coords[1,0])), 
          (float(self.coords[0, 1]), float(self.coords[1, 1]))], 10000) )
        

    def __getitem__(self, index):
        'Generate one batch of data'
        self.key, subkey = random.split(self.key)
        inputs = self.__data_generation(subkey)
        return inputs

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key):
        'Generates data containing batch_size samples'
        idx = random.choice(key, 10000, (self.batch_size,), replace=False)
        x = self.xvals[idx,:]
        
        return x

class DataGenerator_res_MF2(data.Dataset):
    def __init__(self, dim, res_pts, coords, err_norm,
                 batch_size=64, batch_size_res=32, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.dim = dim
        self.res_pts = res_pts
        self.N = res_pts.shape[0]
        self.err_norm = err_norm
        self.coords = coords

        self.batch_size = batch_size

        self.batch_size_res = batch_size_res
        self.key = rng_key

    def __getitem__(self, index):
        'Generate one batch of data'
        self.key, subkey = random.split(self.key)
        inputs = self.__data_generation(subkey)
        return inputs

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key):
        'Generates data containing batch_size samples'


        batch_1 = self.batch_size_res
        batch_2 = self.batch_size-batch_1
        idx = random.choice(key, self.N, (batch_1,), replace=False, p=self.err_norm)
        key, subkey = random.split(key)
        ax1 = self.res_pts[idx,:]
        
        ax2 = self.coords[0:1,:] + (self.coords[1:2,:]-
                                    self.coords[0:1,:])*\
            random.uniform(key, shape=[batch_2, self.dim])
     

       
        x = np.concatenate([ax1, ax2])

        return x
    
    

# Define the exact solution and its derivatives
def u0(x):
    """
    :param x: x = (t, x)
    """
    t = x[:,0:1]
    x = x[:,1:2]
    return x*x*np.cos(np.pi * x) 
                
# =============================================
# =============================================
    
if __name__ == "__main__":

    ymin_A = float(sys.argv[1])
    ymin_B = float(sys.argv[2])
    init_lr = float(sys.argv[3]) # try 1e-2, 1e-3, 1e-4 
    decay = float(sys.argv[4]) # with decay rates 0.95 and 0.99
    N_nl = int(sys.argv[5]) # try 60 and 80

    # ymin_A = 0.0
    # ymin_B = 1.0
    # init_lr = 1e-3
    # decay = 0.99
    # N_nl = 100
    
    batch_size = 500 #500
    batch_size_res = 0
    # batch_size_res = int(batch_size/2)
    Npts = 5000

    N_low =200
    layers = [2, N_low, N_low, N_low, N_low, N_low, 1]
    layer_sizes_nl = [3, N_nl, N_nl, N_nl, 1]
    # layer_sizes_l = [1, 20, 1]
    layer_sizes_l = [1, 1]


    batch_size_s = 100
    epochs = 100000
    epochsA= 100000
    lr = optimizers.exponential_decay(init_lr, decay_steps=2000, decay_rate=decay)
    lrA = optimizers.exponential_decay(1e-4, decay_steps=2000, decay_rate=0.99)
    ics_weight = 1.0 # was 10
    res_weight = 1.0
    ut_weight = 1.0
    energy_weight = 0. # I changed this to 0 from 1 because I don't know what c and a are

    #==== parameters that I am adding =====
    delta = 1.9
    #======================================
    
    steps_to_train = np.arange(3)
    reload = [False, False, False]
    
    reloadA = False

    c = 1. # I put this in so the code will run. The energy weight is 0, so it doesn't matter.
    a = 1. # I put this in so the code will run. The energy weight is 0, so it doesn't matter.
    E_0 = 1/4*math.pi**2*c**2*(1 + 16*a**2) 

    N = 100

    save_str = "MF_loop_res10"

    # path_to_AC = "C:/Users/beec613/Desktop/pnnl_research/code/damiens_code/good_code/allen_cahn_dd/onet_scripts/"
    # path = "C:/Users/beec613/Desktop/pnnl_research/code/damiens_code/good_code/allen_cahn_dd/out_results/cahn_" + str(int(ymin_A)) + "_" + str(ymin_B) + "_" + str(init_lr) + "_" + str(decay) + "_" + str(N_nl)
    
    path_to_AC = "/people/beec613/pnnl_research/code/damiens_code/good_code/allen_cahn_dd/onet_scripts/"
    path = "/people/beec613/pnnl_research/code/damiens_code/good_code/allen_cahn_dd/out_results/cahn_" + str(int(ymin_A)) + "_" + str(ymin_B) + "_" + str(init_lr) + "_" + str(decay) + "_" + str(N_nl)
    
    # ====================================
    # saving settings
    # ====================================
    results_dir_A = path + "/results_A/"+save_str+"/"
    if not os.path.exists(results_dir_A):
        os.makedirs(results_dir_A)
        
    # ====================================
    # DNN model A
    # ====================================
    model_A = DNN_class(layers, ics_weight, res_weight, ut_weight, lrA, restart =0)

    ics_coords = np.array([[ymin_A, -1.0],[ymin_A, 1.0]])
    bc1_coords = np.array([[ymin_A, -1.0],[ymin_B, -1.0]])
    bc2_coords = np.array([[ymin_A, 1.0],[ymin_B, 1.0]])
    dom_coords = np.array([[ymin_A, -1.0],[ymin_B, 1.0]])

    ics_sampler = DataGenerator_ICS_A(2, ics_coords, lambda x: u0(x), batch_size)
    bc1 = DataGenerator_BCS_A(2, bc1_coords, bc2_coords, batch_size_s)
    bc2 = DataGenerator_BCS_A(2, bc1_coords, bc2_coords, batch_size_s)
    res_sampler = DataGenerator_res_A(2, dom_coords, batch_size)
    # energy_sampler = DataGenerator_energy(2, dom_coords, E_0, N, 10)

    if reloadA:
        params_A = model_A.unravel_params(np.load(results_dir_A + '/params.npy'))
    else:     
        model_A.train(ics_sampler, bc1, bc2, res_sampler, nIter=epochsA, F = 0, lam = [])

        print('\n ... A Training done ...')
        scipy.io.savemat(results_dir_A +"losses.mat", 
                     {'training_loss':model_A.loss_training_log,
                      'res_loss':model_A.loss_res_log,
                      'ics_loss':model_A.loss_ics_log,
                      'ut_loss':model_A.loss_ut_log}, format='4')
                    #   'energy_loss':model_A.loss_energy_log,
    
        params_A = model_A.get_params(model_A.opt_state)
        flat_params, _  = ravel_pytree(model_A.get_params(model_A.opt_state))
        np.save(results_dir_A + 'params.npy', flat_params)
    save_data(model_A, params_A, results_dir_A, path_to_AC)


    # ====================================
    # DNN model A2
    # ====================================
    params_prev = []


    rng_key = random.PRNGKey(1234)
    key, subkey = random.split(rng_key)
    res_pts = dom_coords[0:1,:] + (dom_coords[1:2,:]-dom_coords[0:1,:])*random.uniform(key, shape=[10000, 2])
    x, err_norm = get_pts(model_A, params_A, dom_coords,  N = Npts)
    res_sampler = DataGenerator_res_MF2(2, x, dom_coords, err_norm, batch_size, batch_size_res)

    Ndomains = []
    for step in steps_to_train:
        results_dir =  path + "/results_" + str(step) + "/"+save_str+"/"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        if step == 0:
            res = 0
        else:
            res = 1

        Ndomains.append(2**(step+1))
        model = MF_DNN_class(layer_sizes_nl, layer_sizes_l,layers, ics_weight, res_weight, ut_weight, lr, Ndomains, delta, dom_coords,
                             params_A, params_t=params_prev, restart =res)
        
    
        
        if reload[step]:
            params = model.unravel_params(np.load(results_dir + '/params.npy'))
        
        else:     
            model.train(ics_sampler, bc1, bc2, res_sampler, nIter=epochs)



            print('\n ... A2 Training done ...')
            scipy.io.savemat(results_dir +"losses.mat", 
                         {'training_loss':model.loss_training_log,
                          'res_loss':model.loss_res_log,
                          'ics_loss':model.loss_ics_log,
                          'ut_loss':model.loss_ut_log}, format='4')
        
            params = model.get_params(model.opt_state)
            flat_params, _  = ravel_pytree(params)
            np.save(results_dir + 'params.npy', flat_params)
        
            save_data_MF(model, params, results_dir, path_to_AC)
            
        params_prev.append(params)
        
        key, subkey = random.split(rng_key)
        res_pts = dom_coords[0:1,:] + (dom_coords[1:2,:]-dom_coords[0:1,:])*random.uniform(key, shape=[10000, 2])
        x, err_norm = get_pts(model, params, dom_coords,  N = Npts)
        res_sampler = DataGenerator_res_MF2(2, x, dom_coords, err_norm, batch_size, batch_size_res)
    
            
      

             
# =============================================
# =============================================
    
#    replay = False
#    MAS = False
#    RDPS = False 
#    scaled = False
#    run_MF(replay, MAS, RDPS, scaled, MASl=0, N=300) #MF



    
    
    
                        
