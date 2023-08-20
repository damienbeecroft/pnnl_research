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
from utils_fs_v2 import timing, DNN, nonlinear_DNN, linear_DNN
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
import numpy as onp

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

# Define the exact solution and its derivatives
def u0(x):
    """
    :param x: x = (t, x)
    """
    t = x[:,0:1]
    x = x[:,1:2]
    return x*x*np.cos(np.pi * x) 


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
        # self.energy_weight = energy_weight


        # building loss function
        self.loss_training_log = []
        self.loss_res_log = []
        self.loss_ics_log = []
        self.loss_ut_log = []
        # self.loss_energy_log = []

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
          s_t = grad(self.operator_net, argnums= 2)(params, x, t)
          s = self.operator_net(params, x, t)
    
          res = s_t - 0.0001*s_xx+5*s**3-5.0*s
          return res
    
    def ux_net(self, params, u):
          x = u[1]
          t = u[0]
          
          s_t = grad(self.operator_net, argnums= 1)(params, x, t)
          return s_t
      
    
    
    # def energy_subnet(self, params, u):
    #     x = u[1]
    #     t = u[0]


    #     s_x = grad(self.operator_net, argnums= 1)(params, x, t)
    #     s_t = grad(self.operator_net, argnums= 2)(params, x, t)


    #     return s_x, s_t
    
    
#     #define energy net
#     def energy_net(self, params, u, dx):


#         s_x, s_t = vmap(self.energy_subnet, (None, 0))(params, u)
#         s = vmap(self.operator_net, (None, 0))(params, u)

#         integrand = 0.5*s_x**2 + s_t**2 + 0.25-0.5*s**2+0.25*s**4
#  #       integrand = np.expand_dims(integrand, axis=1)
#      #   integral = np.trapz(integrand, dx)
#         integral = dx/2*(2*np.sum(integrand)-integrand[0]-integrand[-1])
     
#        # integral= dx*np.sum(integrand)
#         return integral
    
    # def loss_energy(self, params, batch):
    #     # Fetch data
    #     inputs, outputs = batch
    #     u, dx = inputs


        # # Compute forward pass
        # res1_pred  = vmap(self.energy_net, (None, 0,
        #                                     None))(params, u, dx)
        # loss_en = np.mean((res1_pred-outputs)**2)

        # return loss_en   
    
    
    
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
    
    
      
      # Define residual loss
    def loss_res(self, params, batch):
          # Fetch data
          inputs = batch
          u = inputs
    
          # Compute forward pass
          res1_pred  = vmap(self.residual_net, (None, 0))(params, u)
          loss_res = np.mean((res1_pred)**2)
          return loss_res   
    
    def loss_bcs(self, params, batch):
          # Fetch data
          inputs = batch
          u1, u2 = inputs
          x1 = u1[:, 1]
          t1 = u1[:, 0]
          x2 = u2[:, 1]
          t2 = u2[:, 0]
          
          # Compute forward pass
          s1_pred =vmap(self.operator_net, (None, 0, 0))(params, x1, t1)
          s2_pred =vmap(self.operator_net, (None, 0, 0))(params, x2, t2)
    
          # Compute loss
          loss_s_bc = np.mean((s1_pred - s2_pred)**2)
    
          return loss_s_bc
    
    def loss_bcs_x(self, params, batch):
          # Fetch data
          inputs = batch
          u1, u2 = inputs
    
          # Compute forward pass
    
          s_x_bc1_pred = vmap(self.ux_net, (None, 0))(params, u1)
          s_x_bc2_pred = vmap(self.ux_net, (None, 0))(params, u2)
    
          # Compute loss
          loss_s_x_bc = np.mean((s_x_bc1_pred - s_x_bc2_pred)**2)
    
          return loss_s_x_bc  
      
    
    
    # Define total loss
    def loss(self, params, params_t, ics_batch, bc1_batch, bc2_batch, res_batch, F, lam):
        loss_ics = self.loss_ics(params, ics_batch)
        loss_bcs = self.loss_bcs(params, bc1_batch)
        loss_bcs_x = self.loss_bcs_x(params, bc2_batch)
        loss_res = self.loss_res(params, res_batch)
        # loss_energy = self.loss_energy(params, en_batch)


        loss =  10.0*self.ics_weight*(loss_ics)\
                + self.res_weight*loss_res \
                + self.ut_weight*(loss_bcs + loss_bcs_x)
                # + self.energy_weight*loss_energy
        
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
        # energy_data = iter(energy_dataset)

        pbar = trange(nIter)
        # Main training loop
        for it in pbar:
            # Fetch data
            res_batch= next(res_data)
            ics_batch= next(ics_data)
            bc1_batch= next(bc1_data)
            bc2_batch= next(bc2_data)
            # energy_batch = next(energy_data)

            self.opt_state = self.step(next(self.itercount), self.opt_state, self.params_t, ics_batch, bc1_batch, bc2_batch, res_batch, 
                                        F, lam)
            
            if it % 1000 == 0:
                params = self.get_params(self.opt_state)

                # Compute losses
                loss_value = self.loss(params, self.params_t, ics_batch, bc1_batch, bc2_batch, res_batch, F, lam)
                
                res_value = self.loss_res(params, res_batch)
                ics_value = self.loss_ics(params, ics_batch)
                bcs_value = self.loss_bcs(params, bc1_batch)+self.loss_bcs_x(params, bc2_batch)
                # energy_value = self.loss_energy(params, energy_batch)

                # Store losses
                self.loss_training_log.append(loss_value)
                self.loss_res_log.append(res_value)
                self.loss_ics_log.append(ics_value)
                self.loss_ut_log.append(bcs_value)
                # self.loss_energy_log.append(energy_value)

                # Print losses
                pbar.set_postfix({'Loss': "{0:.4f}".format(loss_value), 
                                  'Res': "{0:.4f}".format(res_value), 
                                  'ICS': "{0:.4f}".format(ics_value),
                                  'BCS': "{0:.4f}".format(bcs_value)})

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
    def predict_res(self, params, U_star):
        s_pred =vmap(self.residual_net, (None, 0))(params, U_star)
        return s_pred
    


class MF_DNN_class:
    
    # Initialize the class
    def __init__(self, layers_branch_nl, layers_branch_l,  layers_lf, ics_weight, res_weight, 
                 ut_weight, lr , params_A, restart =0, params_t = 0): 

        self.init_nl, self.apply_nl, self.weight_nl = nonlinear_DNN(layers_branch_nl)
        self.init_l, self.apply_l = linear_DNN(layers_branch_l)

        self.init_lf, self.apply_lf = DNN(layers_lf)


        self.params_A = params_A
        self.params_t = params_t


        if restart == 1 and len(self.params_t) > 0:
            params_nl = self.params_t[-1][0]
            params_l = self.params_t[-1][1]
        else:
            params_nl = self.init_nl(random.PRNGKey(13))
            params_l = self.init_l(random.PRNGKey(12345))
        params = (params_nl, params_l)
        self.restart = restart

        
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
        y = y.reshape([-1, 2])
        ul = self.apply_lf(self.params_A, y)[0, 0]

        for i in onp.arange(len(self.params_t)): 
            paramsB_nl =  self.params_t[i][0]
            paramsB_l =  self.params_t[i][1]
            y = np.stack([t,x, ul])

         #   in_u = np.hstack([y, ul])

            B_lin = self.apply_l(paramsB_l, ul)
           # B_lin = self.apply_l(paramsB_l,in_u)
            B_nonlin = self.apply_nl(paramsB_nl, y)
            B_lin = B_lin[:, 0]

            ul = B_nonlin + B_lin 
            ul = ul[0]
        
        params_nl, params_l = params
        y = np.stack([t,x, ul])

        logits_nl = self.apply_nl(params_nl, y)
        logits_l = self.apply_l(params_l, ul)
        logits_l = logits_l[:, 0]
        pred = logits_nl + logits_l 

        
        return pred[0]
    
    
    
    def operator_net_nl(self, params, x, t):
        y = np.stack([t,x])
        y = y.reshape([-1, 2])
        ul = self.apply_lf(self.params_A, y)[0, 0]

        for i in onp.arange(len(self.params_t)): 
            paramsB_nl =  self.params_t[i][0]
            paramsB_l =  self.params_t[i][1]
            y = np.stack([t,x, ul])

         #   in_u = np.hstack([y, ul])

            B_lin = self.apply_l(paramsB_l, ul)
           # B_lin = self.apply_l(paramsB_l,in_u)
            B_nonlin = self.apply_nl(paramsB_nl, y)
            B_lin = B_lin[:, 0]

            ul = B_nonlin + B_lin 
            ul = ul[0]
        
        params_nl, params_l = params
        y = np.stack([t,x, ul])

        logits_nl = self.apply_nl(params_nl, y)

        pred = logits_nl  

        
        return pred[0]
    
    
    
    def operator_net_l(self, params, x, t):
        y = np.stack([t,x])
        y = y.reshape([-1, 2])
        ul = self.apply_lf(self.params_A, y)[0, 0]

        for i in onp.arange(len(self.params_t)): 
            paramsB_nl =  self.params_t[i][0]
            paramsB_l =  self.params_t[i][1]
            y = np.stack([t,x, ul])

         #   in_u = np.hstack([y, ul])

            B_lin = self.apply_l(paramsB_l, ul)
           # B_lin = self.apply_l(paramsB_l,in_u)
            B_nonlin = self.apply_nl(paramsB_nl, y)
            B_lin = B_lin[:, 0]

            ul = B_nonlin + B_lin 
            ul = ul[0]
        
        params_nl, params_l = params
        y = np.stack([t,x, ul])

        logits_l = self.apply_l(params_l, ul)
        logits_l = logits_l[:, 0]
        pred = logits_l 
        
        return pred[0]
    


    def residual_net(self, params, x, t):

        
        s_xx = grad(grad(self.operator_net, argnums= 1), argnums= 1)(params, x, t)
        s_t = grad(self.operator_net, argnums= 2)(params, x, t)
        s = self.operator_net(params, x, t)

        res = s_t - 0.0001*s_xx+5*s**3-5.0*s
        return res

    def ux_net(self, params, u):
        x = u[1]
        t = u[0]
        
        s_t = grad(self.operator_net, argnums= 1)(params, x, t)
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

    def loss_bcs(self, params, batch):
          # Fetch data
          inputs = batch
          u1, u2 = inputs
          x1 = u1[:, 1]
          t1 = u1[:, 0]
          x2 = u2[:, 1]
          t2 = u2[:, 0]
          
          # Compute forward pass
          s1_pred =vmap(self.operator_net, (None, 0, 0))(params, x1, t1)
          s2_pred =vmap(self.operator_net, (None, 0, 0))(params, x2, t2)
    
          # Compute loss
          loss_s_bc = np.mean((s1_pred - s2_pred)**2)
    
          return loss_s_bc

    def loss_bcs_x(self, params, batch):
          # Fetch data
          inputs = batch
          u1, u2= inputs

          # Compute forward pass
    
          s_x_bc1_pred = vmap(self.ux_net, (None, 0))(params, u1)
          s_x_bc2_pred = vmap(self.ux_net, (None, 0))(params, u2)
    
          # Compute loss
          loss_s_x_bc = np.mean((s_x_bc1_pred - s_x_bc2_pred)**2)
    
          return loss_s_x_bc  
    
    # Define residual loss
    def loss_res(self, params, batch):
        # Fetch data
        inputs = batch
        u = inputs
        x = u[:, 1]
        t = u[:, 0]
        
        # Compute forward pass
        res1_pred  = vmap(self.residual_net, (None, 0, 0))(params, x, t)
        loss_res = np.mean((res1_pred)**2)
        return loss_res   

    
    # Define total loss
    @partial(jit, static_argnums=(0,))
    def loss(self, params, ics_batch, bc1_batch, bc2_batch, res_batch):
        loss_ics = self.loss_ics(params, ics_batch)
        loss_bc1 = self.loss_bcs(params, bc1_batch)
        loss_bcs_x = self.loss_bcs_x(params, bc2_batch)
        loss_res = self.loss_res(params, res_batch)


        loss =  self.ics_weight*(loss_ics)\
                + self.res_weight*loss_res \
                + self.ut_weight*(loss_bc1+loss_bcs_x)
                
        params_nl, params_l = params

        loss += 1.0e-5*self.weight_nl(params_nl)
        return loss

    @partial(jit, static_argnums=(0,))
    def loss_ICS_full(self, params, ics_batch, bc1_batch, bc2_batch, res_batch):
        loss_ics = self.loss_ics(params, ics_batch)

        loss =  loss_ics
        params_nl, params_l = params

        loss += 1.0e-5*self.weight_nl(params_nl)
        return loss
        
        
        # Define a compiled update step
    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, ics_batch, bc1_batch, bc2_batch, res_batch):
        params = self.get_params(opt_state)

        g = grad(self.loss)(params, ics_batch, bc1_batch, bc2_batch, res_batch)
        return self.opt_update(i, g, opt_state)

        # Define a compiled update step
    @partial(jit, static_argnums=(0,))
    def step_ICS(self, i, opt_state, ics_batch, bc1_batch, bc2_batch, res_batch):
        params = self.get_params(opt_state)

        g = grad(self.loss_ICS_full)(params,ics_batch, bc1_batch, bc2_batch, res_batch)
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

            self.opt_state = self.step(next(self.itercount), self.opt_state,
                                       ics_batch, bc1_batch, bc2_batch, res_batch)
            
            if it % 1000 == 0:
                params = self.get_params(self.opt_state)

                # Compute losses
                loss_value = self.loss(params, ics_batch, bc1_batch, bc2_batch, res_batch)
                
                res_value = self.loss_res(params, res_batch)
                ics_value = self.loss_ics(params, ics_batch)
                bcs_value = self.loss_bcs(params, bc1_batch)+self.loss_bcs_x(params, bc2_batch)

              #  print(res_value.shape)
              #  print(ics_value.shape)
                # Store losses
                self.loss_training_log.append(loss_value)
                self.loss_res_log.append(res_value)
                self.loss_ics_log.append(ics_value)
                self.loss_ut_log.append(bcs_value)

                # Print losses
                pbar.set_postfix({'Loss': loss_value, 
                                  'Res': res_value, 
                                  'ICS':ics_value,
                                  'BCS': bcs_value})

                

    # Evaluates predictions at test points  
    def predict_u(self, params, U_star):
        x = U_star[:, 1]
        t = U_star[:, 0]
        
      #  s_pred = vmap(self.apply_low, (None, 0, 0))(params_low, U_star, Y_star)
        s_pred =vmap(self.operator_net, (None, 0, 0))(params, x, t)
        return s_pred


    def predict_u_nl(self, params, U_star):
        x = U_star[:, 1]
        t = U_star[:, 0]
        
      #  s_pred = vmap(self.apply_low, (None, 0, 0))(params_low, U_star, Y_star)
        s_pred =vmap(self.operator_net_nl, (None, 0, 0))(params, x, t)
        return s_pred


    def predict_u_l(self, params, U_star):
        x = U_star[:, 1]
        t = U_star[:, 0]
        
      #  s_pred = vmap(self.apply_low, (None, 0, 0))(params_low, U_star, Y_star)
        s_pred =vmap(self.operator_net_l, (None, 0, 0))(params, x, t)
        return s_pred

    def predict_res(self, params, U_star):

        x = U_star[:, 1]
        t = U_star[:, 0]
        
      #  s_pred = vmap(self.apply_low, (None, 0, 0))(params_low, U_star, Y_star)
        s_pred =vmap(self.residual_net, (None, 0, 0))(params,  x, t)
        return s_pred
    
    def predict_ut(self, params, U_star):
        s_pred =vmap(self.ut_net, (None, 0))(params, U_star)
        return s_pred
    

