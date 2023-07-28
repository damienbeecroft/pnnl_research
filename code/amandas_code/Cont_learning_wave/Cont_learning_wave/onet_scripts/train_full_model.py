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
import pandas as pd
from scipy.interpolate import griddata
#import matplotlib
#import math
#import matplotlib.pyplot as plt
#import numpy as np

######################################################################
#######################  Standard DeepONets ##########################
######################################################################

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
    def __init__(self, layers_branch_low, ics_weight, res_weight, ut_weight, lr): 


        #Network initialization 
        self.init_low, self.apply_low = DNN(layers_branch_low, activation=np.tanh)
        params_low = self.init_low(random.PRNGKey(1))
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

        # Compute forward pass
        res1_pred  = vmap(self.ut_net, (None, 0))(params, u)
        loss_res = np.mean((res1_pred)**2)
        return loss_res   
    
    
    # Define total loss
    def loss(self, params, ics_batch, bc1_batch, bc2_batch, res_batch):
        loss_ics = self.loss_ics(params, ics_batch)
        loss_bc1 = self.loss_ics(params, bc1_batch)
        loss_bc2 = self.loss_ics(params, bc2_batch)
        loss_ut = self.loss_ut(params, ics_batch)
        loss_res = self.loss_res(params, res_batch)


        loss =  10.0*self.ics_weight*(loss_ics+loss_bc1+loss_bc2)\
                + self.res_weight*loss_res \
                + self.ut_weight*loss_ut
        return loss
    

    
        # Define a compiled update step
    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, ics_batch, bc1_batch, bc2_batch, res_batch):
        params = self.get_params(opt_state)

        g = grad(self.loss)(params, ics_batch, bc1_batch, bc2_batch, res_batch)
        return self.opt_update(i, g, opt_state)
    

    # Optimize parameters in a loop
    def train(self, ics_dataset, bc1_dataset, bc2_dataset, res_dataset, nIter = 10000):
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

            self.opt_state = self.step(next(self.itercount), self.opt_state, ics_batch, bc1_batch, bc2_batch, res_batch)
            
            if it % 1000 == 0:
                params = self.get_params(self.opt_state)

                # Compute losses
                loss_value = self.loss(params, ics_batch, bc1_batch, bc2_batch, res_batch)
                
                res_value = self.loss_res(params, res_batch)
                ics_value = self.loss_ics(params, ics_batch)+self.loss_ics(params, bc1_batch)+self.loss_ics(params, bc2_batch)
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
        params_low = params
        x = U_star[:, 1]
        t = U_star[:, 0]
        
      #  s_pred = vmap(self.apply_low, (None, 0, 0))(params_low, U_star, Y_star)
        s_pred =vmap(self.operator_net, (None, 0, 0))(params, x, t)
        return s_pred

                    
# =============================================
# =============================================
    
if __name__ == "__main__":


    N_low = 100
    layers_branch_low = [2, N_low, N_low, N_low, N_low, N_low, 1]
    a = 0.5
    c = 2
    batch_size = 100
    epochs = 80000
    lr = optimizers.exponential_decay(1e-3, decay_steps=2000, decay_rate=0.9)

    ymin = 0.0
    ymax = 0.2
    ics_coords = np.array([[0.0, 0.0],
                        [0.0, 1.0]])
    bc1_coords = np.array([[0.0, 0.0],
                            [ymax, 0.0]])
    bc2_coords = np.array([[0.0, 1.0],
                            [ymax, 1.0]])
    dom_coords = np.array([[0.0, 0.0],
                            [ymax, 1.0]])

    # Create initial conditions samplers
    ics_sampler = DataGenerator(2, ics_coords, lambda x: u(x, a, c), batch_size)
    
    # Create boundary conditions samplers
    bc1 = DataGenerator(2, bc1_coords, lambda x: u(x, a, c), 20)
    bc2 = DataGenerator(2, bc2_coords, lambda x: u(x, a, c), 20)
    bcs_sampler = [bc1, bc2]
    res_sampler = DataGenerator(2, dom_coords, lambda x: r(x, a, c), batch_size)

    
   # d_vx = scipy.io.loadmat("../data.mat")
   # t_data, s_data = (d_vx["u"].astype(np.float32), 
   #            d_vx["s"].astype(np.float32))
   # t_data = t_data[:, data_range].reshape([-1, 1])
   # s_data = s_data[data_range, :].reshape([-1, 2])
   # t_data = jax.device_put(t_data)
   # s_data = jax.device_put(s_data)
   # data_dataset = DataGenerator(t_data, s_data, 10)

    save_str = "full_training"

    # ====================================
    # saving settings
    # ====================================
    results_dir = "../results_full/"+save_str+"/"
    save_results_to = results_dir
    if not os.path.exists(save_results_to):
        os.makedirs(save_results_to)

    # ====================================
    # deeponet model
    # ====================================
    ics_weight = 1
    res_weight = 1
    ut_weight = 1

    model = DNN_class(layers_branch_low, ics_weight, res_weight, ut_weight, lr)
                    
    model.train(ics_sampler, bc1, bc2, res_sampler, nIter=epochs)
    print('\n ... Training done ...')
    
    scipy.io.savemat(save_results_to +"losses.mat", 
                     {'training_loss':model.loss_training_log,
                      'res_loss':model.loss_res_log,
                      'ics_loss':model.loss_ics_log,
                      'ut_loss':model.loss_ut_log})
    
    # ====================================
    # testing
    # ====================================

    
    if 1:
        params = model.get_params(model.opt_state)
        flat_params, _  = ravel_pytree(model.get_params(model.opt_state))
        np.save(results_dir + 'params_A.npy', flat_params)


        nn = 50
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
        


