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
from utils_fs_v2 import timing, DNN
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
from copy import deepcopy


######################################################################
#######################  Standard DeepONets ##########################
######################################################################

def gen_prev_data_B(model_A, u, params_A):
    predA = model_A.predict_u(params_A, u)
    return predA



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
    def __init__(self, dim, coords, model, params, 
                 batch_size=64, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.dim = dim
        self.coords = coords
        self.params = params
        self.model = model
        
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
        y = self.model.predict_u(self.params, x)
        yt = self.model.predict_ut(self.params, x)
        inputs = x
        outputs = (y, yt)
        return inputs, outputs
    

class DataGenerator_ICS_A(data.Dataset):
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

class DNN_class:
    
    # Initialize the class
    def __init__(self, layers_branch_low, ics_weight, res_weight, ut_weight, lr , restart =0, params_t = 0, params_i = 0): 


        #Network initialization 
        self.init_low, self.apply_low = DNN(layers_branch_low, activation=np.tanh)
        self.params_t = params_t
        if restart ==1:
            params_low = params_i
        else:
            params_low = self.init_low(random.PRNGKey(10))
        params = (params_low)

        
        
        # Use optimizers to set optimizer initialization and update functions
        self.opt_init, \
        self.opt_update, \
        self.get_params = optimizers.adam(lr)
     
        self.opt_state = self.opt_init(params)

        # Used to restore the trained model parameters
        _, self.unravel_params = ravel_pytree(params)

        self.itercount = itertools.count()


        self.ics_weight = ics_weight
        self.res_weight = res_weight
        self.ut_weight = ut_weight


        # building loss function
        self.loss_training_log = []
        self.loss_res_log = []
        self.loss_ics_log = []
        self.loss_ut_log = []

    # =============================================
    # evaluation
    # =============================================

    # Define DeepONet architecture
    def operator_net(self, params, x, t):
        y = np.stack([t,x])
        B = self.apply_low(params, y)
        return B[0]

    # Define ODE residual
    def residual_net(self, params, u):
        x = u[1]
        t = u[0]
        
        s_xx = grad(grad(self.operator_net, argnums= 1), argnums= 1)(params, x, t)
        s_tt = grad(grad(self.operator_net, argnums= 2), argnums= 2)(params, x, t)

        res = s_tt - 4*s_xx
        return res

    def ut_net(self, params, u):
        x = u[1]
        t = u[0]
        
        s_t = grad(self.operator_net, argnums= 2)(params, x, t)
        return s_t
    
    
    # Define initial loss
    def loss_ics(self, params, batch):
        # Fetch data
        inputs, outputs = batch
        u = inputs
        x = u[:, 1]
        t = u[:, 0]
        
        
        s1, s1t = outputs

        # Compute forward pass
        s1_pred =vmap(self.operator_net, (None, 0, 0))(params, x, t)
        # Compute loss

        loss = np.mean((s1.flatten() - s1_pred.flatten())**2)

        return loss

    def loss_bcs(self, params, batch):
        # Fetch data
        inputs, outputs = batch
        u = inputs
        x = u[:, 1]
        t = u[:, 0]
        
        
        s1 = outputs

        # Compute forward pass
        s1_pred =vmap(self.operator_net, (None, 0, 0))(params, x, t)
        # Compute loss

        loss = np.mean((s1.flatten() - s1_pred.flatten())**2)

        return loss
    
    def loss_data(self, params, batch):
        # Fetch data
        inputs, outputs = batch
        u = inputs
        
        s1 = outputs[:, 0:1]
        s2 = outputs[:, 1:2]

        # Compute forward pass
        s1_pred, s2_pred =vmap(self.operator_net, (None, 0))(params, u)
        # Compute loss

        loss_s1 = np.mean((s1.flatten() - s1_pred.flatten())**2)
        loss_s2 = np.mean((s2.flatten() - s2_pred.flatten())**2)

        loss = loss_s1 + loss_s2
        return loss
    
    # Define residual loss
    def loss_res(self, params, batch):
        # Fetch data
        inputs, outputs = batch
        u = inputs

        # Compute forward pass
        res1_pred  = vmap(self.residual_net, (None, 0))(params, u)
        loss_res = np.mean((res1_pred)**2)
        return loss_res   

    def loss_ut(self, params, batch):
        # Fetch data
        inputs, outputs = batch
        u = inputs
        s1, s1t = outputs

        # Compute forward pass
        res1_pred  = vmap(self.ut_net, (None, 0))(params, u)
        loss_res = np.mean((res1_pred.flatten()-s1t.flatten())**2)
        return loss_res   
    
    
    # Define total loss
    def loss(self, params, params_t, ics_batch, bc1_batch, bc2_batch, res_batch, F, lam):
        loss_ics = self.loss_ics(params, ics_batch)
        loss_bc1 = self.loss_bcs(params, bc1_batch)
        loss_bc2 = self.loss_bcs(params, bc2_batch)
        loss_ut = self.loss_ut(params, ics_batch)
        loss_res = self.loss_res(params, res_batch)


        loss =  self.ics_weight*(loss_ics+loss_bc1+loss_bc2)\
                + self.res_weight*loss_res \
                + self.ut_weight*loss_ut
                
        count = 0
        s = 0.0
        n = len(params)

        for j in range(len(lam)):
            for k in range(len(params)):
                s += lam[j]/2 * np.sum(F[count]*(params[k][0]-params_t[n*j + k][0])**2)
                count += 1
                s += lam[j]/2 * np.sum(F[count]*(params[k][1]-params_t[n*j + k][1])**2)
                count += 1
        loss += s
                
        return loss

    
        # Define a compiled update step
    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, params_t, ics_batch, bc1_batch, bc2_batch, res_batch, F, lam):
        params = self.get_params(opt_state)

        g = grad(self.loss)(params, params_t, ics_batch, bc1_batch, bc2_batch, res_batch, F, lam)
        return self.opt_update(i, g, opt_state)
    

    # Optimize parameters in a loop
    def train(self, ics_dataset, bc1_dataset, bc2_dataset, res_dataset, nIter = 10000, F = 0, lam = []):
        res_data = iter(res_dataset)
        ics_data = iter(ics_dataset)
        bc1_data = iter(bc1_dataset)
        bc2_data = iter(bc2_dataset)

        pbar = trange(nIter)
        # Main training loop
        for it in pbar:
            # Fetch data
            res_batch= next(res_data)
            ics_batch= next(ics_data)
            bc1_batch= next(bc1_data)
            bc2_batch= next(bc2_data)

            self.opt_state = self.step(next(self.itercount), self.opt_state, self.params_t, ics_batch, bc1_batch, bc2_batch, res_batch, F, lam)
            
            if it % 1000 == 0:
                params = self.get_params(self.opt_state)

                # Compute losses
                loss_value = self.loss(params, self.params_t, ics_batch, bc1_batch, bc2_batch, res_batch, F, lam)
                
                res_value = self.loss_res(params, res_batch)
                ics_value = self.loss_ics(params, ics_batch)+self.loss_bcs(params, bc1_batch)+self.loss_bcs(params, bc2_batch)
                ut_value = self.loss_ut(params, ics_batch)

                # Store losses
                self.loss_training_log.append(loss_value)
                self.loss_res_log.append(res_value)
                self.loss_ics_log.append(ics_value)
                self.loss_ut_log.append(ut_value)

                # Print losses
                pbar.set_postfix({'Loss': "{0:.4f}".format(loss_value), 
                                  'Res': "{0:.4f}".format(res_value), 
                                  'ICS': "{0:.4f}".format(ics_value),
                                  'Ut': "{0:.4f}".format(ut_value)})

    # Evaluates predictions at test points  
  #  @partial(jit, static_argnums=(0,))
    def predict_u(self, params, U_star):
        x = U_star[:, 1]
        t = U_star[:, 0]
        
      #  s_pred = vmap(self.apply_low, (None, 0, 0))(params_low, U_star, Y_star)
        s_pred =vmap(self.operator_net, (None, 0, 0))(params, x, t)
        return s_pred

    def predict_ut(self, params, U_star):
        s_pred =vmap(self.ut_net, (None, 0))(params, U_star)
        return s_pred
    
    
    @partial(jit, static_argnums=(0,))
    def predict_log(self, params, U_star):
        pred = self.predict_u(params, U_star)

        return np.log(pred[0])
    
    
    def compute_fisher(self, params, coords, key,  num_samples=200, plot_diffs=True, disp_freq=1):
    
        branch_layers = len(params)
        # initialize Fisher information for most recent task
        F_accum = []
        for k in range(branch_layers):
            F_accum.append(np.zeros(params[k][0].shape))
            F_accum.append(np.zeros(params[k][1].shape))
    
        if(plot_diffs):
            # track differences in mean Fisher info
            F_prev = deepcopy(F_accum)
            mean_diffs = np.zeros(0)
    
        for i in range(num_samples):
            # select random input image
            key, subkey = random.split(key)
    
            u = coords[0:1,:] + (coords[1:2,:]- coords[0:1,:])*random.uniform(key, shape=[1, 2])
            #idx = random.choice(subkey, u.shape[0], (1,), replace=False)
          #  print(idx)
            ders = grad(self.predict_log)(params, u)
            for k in range(branch_layers):
                F_accum[2*k] += np.square(ders[k][0])
                F_accum[2*k+1] += np.square(ders[k][1])
    
            if(plot_diffs):
                if i % disp_freq == 0 and i > 0:
                    # recording mean diffs of F
                    F_diff = 0
                    for v in range(len(F_accum)):
                        F_diff += np.sum(np.absolute(F_accum[v]/(i+1) - F_prev[v]))
                    mean_diff = np.mean(F_diff)
                    mean_diffs = np.append(mean_diffs, mean_diff)
                    print(mean_diff)

                    for v in range(len(F_accum)):
                        F_prev[v] = F_accum[v]/(i+1)
                        
                        
 #       i = num_samples -1
  #      plt.semilogy(range(disp_freq+1, i+2, disp_freq), mean_diffs)
   #     plt.xlabel("Number of samples")
    #    plt.ylabel("Mean absolute Fisher difference")
    
        # divide totals by number of samples
    
        for v in range(len(F_accum)):
            F_accum[v] /= (num_samples)
         #   print(F_accum[v])
        
        return F_accum


                    