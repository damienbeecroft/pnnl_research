"""
Created on July 2021
@author: Qizhi He (qizhi.he@pnnl.gov)
Note: 
* For multiple beta cases
* For GPU runing
* Add save log_loss during training
* <2021.07.01> Modified the DeepOnet structure based on Xuhui Meng's code ["fs_v2"]
"""


# My imports
from jax import config
# config.update("jax_debug_nans", True)
# config.parse_flags_with_jax = False

# Other imports
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from utils_fs_v2 import nonlinear_DNN, linear_DNN, DNN
import math
import jax.numpy as jnp
from jax import random, grad, vmap, jit
from jax.example_libraries import optimizers
from jax import lax
from jax.flatten_util import ravel_pytree

import itertools
from functools import partial
from tqdm import trange

from jax import debug

class MF_class_EWC:
    

    def __init__(self, layers_branch_nl, layers_branch_l, layers_branch_lf, ics_weight, res_weight, data_weight, pen_weight, lr ,
                 Ndomains, delta, Tmax, params_A, restart = 0, params_t = []): 


        self.init_lf, self.apply_lf = DNN(layers_branch_lf)
        self.params_t = params_t


        self.Ndomains = Ndomains
        self.init_nl, self.apply_nl, self.weight_nl = nonlinear_DNN(layers_branch_nl)
        self.init_l, self.apply_l = linear_DNN(layers_branch_l)
        params = []
        for i in jnp.arange(self.Ndomains[-1]):
            if restart == 0:
                params_nl = self.init_nl(random.PRNGKey(13*(i+1)))
                params_l = self.init_l(random.PRNGKey(12345*(i+1)))
            else:
                params_nl = self.params_t[-1][0][0]
                params_l = self.params_t[-1][0][1]
            
            params_cur = (params_nl, params_l)
            params.append(params_cur)

        
        
        self.params_A = params_A

        self.restart = restart
        self.Tmax = Tmax
        self.delta = delta

        self.ics_weight = ics_weight
        self.res_weight = res_weight
        self.data_weight = data_weight
        self.pen_weight = pen_weight
        
        
        
        # Use optimizers to set optimizer initialization and update functions
        self.opt_init, \
        self.opt_update, \
        self.get_params = optimizers.adam(lr)
     
        self.opt_state = self.opt_init(params)

        # Used to restore the trained model parameters
        _, self.unravel_params = ravel_pytree(params)

        self.itercount = itertools.count()



        # building loss function
        self.loss_training_log = []
        self.loss_res_log = []
        self.loss_ics_log = []
        self.loss_data_log = []

    # =============================================
    # evaluation
    # =============================================

    def weight_condition(self,condition,u,mu,sigma):
        w = lax.cond(condition, lambda u: (1 + jnp.cos(math.pi*(u-mu)/sigma))**2, lambda _: 0., u)
        return w
    
    # Changed this function
    def w_jl(self, j, l, u):
        mu = self.Tmax*j/(l-1)
        sigma = self.Tmax*(self.delta/2.0)/(l-1)
        conditions = (u < (mu + sigma)) & (u > (mu - sigma))
        w_jl = vmap(self.weight_condition,(0,0,None,None))(conditions,u,mu,sigma)
        return w_jl
        
        
    def operator_net(self, params, u):
    
        s_lf = self.apply_lf(self.params_A, u)

        j = 0
        for level in self.params_t:
            i = 0
            weight_sum = 0.
            s_lf_cur = 0.
            for mfparams in level:
                paramsB_nl = mfparams[0]
                paramsB_l = mfparams[1]
                y = jnp.hstack([u, s_lf])

                s_nl = self.apply_nl(paramsB_nl, y)
                s_l = self.apply_l(paramsB_l, s_lf)
                
                w = self.w_jl(i, self.Ndomains[j], u)
                weight_sum += w
                s_lf_cur += w*(s_l + s_nl)
                i +=1
            s_lf = s_lf_cur/weight_sum
            j += 1
        
        s1 = 0.
        s2 = 0.
   
        idx = 0
        weight_sum = 0.
        for param in params:
            params_nl, params_l = param
            y = jnp.hstack([u, s_lf])

            s_nl = self.apply_nl(params_nl, y)
            s_l = self.apply_l(params_l, s_lf)
            
            w = self.w_jl(idx, self.Ndomains[-1], u)
            weight_sum += w
            s1 += w*(s_l[:1]+ s_nl[:1])
            s2 += w*(s_l[1:]+ s_nl[1:])
            idx += 1
        s1 = s1/weight_sum
        s2 = s2/weight_sum
        
        return s1, s2
    
    def operator_net_single(self, params, u):
    
        s_lf = self.apply_lf(self.params_A, u)

        j = 0
        for level in self.params_t:
            i = 0
            weight_sum = 0.
            s_lf_cur = 0.
            for mfparams in level:
                paramsB_nl = mfparams[0]
                paramsB_l = mfparams[1]
                y = jnp.hstack([u, s_lf])

                s_nl = self.apply_nl(paramsB_nl, y)
                s_l = self.apply_l(paramsB_l, s_lf)
                
                w = self.w_jl(i, self.Ndomains[j], u)
                weight_sum += w
                s_lf_cur += w*(s_l + s_nl)
                i +=1
            s_lf = s_lf_cur/weight_sum
            j += 1
        
        s1 = 0.
        s2 = 0.
   
        params_nl, params_l = params
        y = jnp.hstack([u, s_lf])

        s_nl = self.apply_nl(params_nl, y)
        s_l = self.apply_l(params_l, s_lf)
        
        s1 += (s_l[:1]+ s_nl[:1])
        s2 += (s_l[1:]+ s_nl[1:])

        #  print(s1.shape)
        
        return s1, s2
    
    def operator_net_multi(self, params, idx, u):
    
        s_lf = self.apply_lf(self.params_A, u)

        j = 0
        for level in self.params_t:
            i = 0
            weight_sum = 0.
            s_lf_cur = 0.
            for mfparams in level:
                paramsB_nl = mfparams[0]
                paramsB_l = mfparams[1]
                y = jnp.hstack([u, s_lf])

                s_nl = self.apply_nl(paramsB_nl, y)
                s_l = self.apply_l(paramsB_l, s_lf)
                
                w = self.w_jl(i, self.Ndomains[j], u)
                weight_sum += w
                s_lf_cur += w*(s_l + s_nl)
                i +=1
            s_lf = s_lf_cur/weight_sum
            j += 1
        
        s1 = 0.
        s2 = 0.
   
        weight_sum = 0.
        for param in params:
            params_nl, params_l = param
            y = jnp.hstack([u, s_lf])

            s_nl = self.apply_nl(params_nl, y)
            s_l = self.apply_l(params_l, s_lf)
            
            w = self.w_jl(idx, self.Ndomains[-1], u)
            weight_sum += w
            s1 += w*(s_l[:1]+ s_nl[:1])
            s2 += w*(s_l[1:]+ s_nl[1:])
            idx += 1
        s1 = s1/weight_sum
        s2 = s2/weight_sum
        
        return s1, s2

    # Define ODE residual
    def residual_net(self, params, u):

        # debug.breakpoint()
        s1, s2 = self.operator_net(params, u)

        def s1_fn(params, u):
          s1_fn, _ = self.operator_net(params, u)
          return s1_fn[0]
        
        def s2_fn(params, u):
          _, s2_fn  = self.operator_net(params, u)
          return s2_fn[0]

        s1_y = grad(s1_fn, argnums= 1)(params, u)
        s2_y = grad(s2_fn, argnums= 1)(params, u)

        res_1 = s1_y - s2
        res_2 = s2_y + 0.05 * s2 + 9.81 * jnp.sin(s1)

        return res_1, res_2

    # Define ODE residual
    def residual_net_single(self, params, u):

        # debug.breakpoint()
        s1, s2 = self.operator_net_single(params, u)

        def s1_fn(params, u):
          s1_fn, _ = self.operator_net_single(params, u)
          return s1_fn[0]
        
        def s2_fn(params, u):
          _, s2_fn  = self.operator_net_single(params, u)
          return s2_fn[0]

        s1_y = grad(s1_fn, argnums= 1)(params, u)
        s2_y = grad(s2_fn, argnums= 1)(params, u)

        res_1 = s1_y - s2
        res_2 = s2_y + 0.05 * s2 + 9.81 * jnp.sin(s1)

        return res_1, res_2
    
    # Define ODE residual
    def residual_net_multi(self, params, idx, u):

        # debug.breakpoint()
        s1, s2 = self.operator_net_multi(params, idx, u)

        def s1_fn(params, idx, u):
          s1_fn, _ = self.operator_net_multi(params, idx, u)
          return s1_fn[0]
        
        def s2_fn(params, idx, u):
          _, s2_fn  = self.operator_net_multi(params, idx, u)
          return s2_fn[0]

        s1_y = grad(s1_fn, argnums= 2)(params, idx, u)
        s2_y = grad(s2_fn, argnums= 2)(params, idx, u)

        res_1 = s1_y - s2
        res_2 = s2_y + 0.05 * s2 + 9.81 * jnp.sin(s1)

        return res_1, res_2
    
    def loss_data(self, params, batch):
        # Fetch data
        inputs, outputs = batch
        u = inputs
        
        s1 = outputs[:, 0:1]
        s2 = outputs[:, 1:2]

        # Compute forward pass
        s1_pred, s2_pred = vmap(self.operator_net, (None, 0))(params, u)
        # Compute loss

        loss_s1 = jnp.mean((s1.flatten() - s1_pred.flatten())**2)
        loss_s2 = jnp.mean((s2.flatten() - s2_pred.flatten())**2)

        loss = loss_s1 + loss_s2
        return loss
    
    # Define residual loss
    def loss_res(self, params, single_res_datasets, double_res_datasets):
        # Fetch data

        res1_pred_sum = 0.
        res2_pred_sum = 0.
        num_u = 0
        # Compute forward pass
        idx = 0
        for batch in single_res_datasets:
            inputs, outputs = batch
            u = inputs
            res1_pred, res2_pred = vmap(self.residual_net_single, (None, 0))(params[idx], u)

            res1_pred_sum += jnp.sum(res1_pred**2)
            res2_pred_sum += jnp.sum(res2_pred**2)
            num_u += len(u)
            idx += 1

        idx = 0
        for batch in double_res_datasets:
            inputs, outputs = batch
            u = inputs
            res1_pred, res2_pred = vmap(self.residual_net_multi, (None, None, 0))([params[idx],params[idx+1]], idx, u)

            res1_pred_sum += jnp.sum(res1_pred**2)
            res2_pred_sum += jnp.sum(res2_pred**2)
            num_u += len(u)
            idx += 1

        # Compute loss

        # NOTE: I hard coded the division by the batch size here. This is temporary. I need to make this adaptable.
        # loss_res = (res1_pred_sum + res2_pred_sum)/100
        loss_res = (res1_pred_sum + res2_pred_sum)/num_u
        return loss_res

    # Define total loss
    def loss(self, params, ic_batch, single_res_datasets, double_res_datasets, val_batch):
        loss_res = self.loss_res(params, single_res_datasets, double_res_datasets)
        loss_ics = self.loss_data(params, ic_batch)
        loss_data = self.loss_data(params, val_batch)
        
        weights  = 0
        for param in params:
             params_nl, params_l = param
             weights += self.weight_nl(params_nl)

        loss =  self.ics_weight*loss_ics + self.res_weight*loss_res +\
            self.data_weight*loss_data+ self.pen_weight*weights
            
        return loss 
    

    
        # Define a compiled update step
    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, ic_batch, single_res_datasets, double_res_datasets, val_batch):
        params = self.get_params(opt_state)

        # g = grad(self.loss)(params, ic_batch, res_batch, val_batch)
        g = grad(self.loss)(params, ic_batch, single_res_datasets, double_res_datasets, val_batch)
        return self.opt_update(i, g, opt_state)
    

    # Optimize parameters in a loop
    # def train(self, ic_dataset, res_dataset, val_dataset, nIter = 10000):
    #     res_data = iter(res_dataset)
    def train(self, ic_dataset, single_res_datasets, double_res_datasets, val_dataset, nIter = 10000):
        # res_data = iter(res_dataset)
        single_res_data = list(map(iter,single_res_datasets))
        double_res_data = list(map(iter,double_res_datasets))

        ic_data = iter(ic_dataset)
        val_data = iter(val_dataset)

        pbar = trange(nIter)
        # Main training loop
        for it in pbar:
            # Fetch data
            single_res_batches = list(map(next,single_res_data))
            double_res_batches = list(map(next,double_res_data))
            ic_batch= next(ic_data)
            val_batch= next(val_data)

            self.opt_state = self.step(next(self.itercount), self.opt_state, 
                                       ic_batch, single_res_batches, double_res_batches, val_batch)
            
            if it % 1000 == 0:
                params = self.get_params(self.opt_state)

                # Compute losses
                loss_value = self.loss(params, ic_batch, single_res_batches, double_res_batches, val_batch)
                res_value = self.loss_res(params, single_res_batches, double_res_batches)
                ics__value = self.loss_data(params, ic_batch)
                data_value = self.loss_data(params, val_batch)

                # Store losses
                self.loss_training_log.append(loss_value)
                self.loss_res_log.append(res_value)
                self.loss_ics_log.append(ics__value)
                self.loss_data_log.append(data_value)

                # Print losses
                pbar.set_postfix({'Loss': "{0:.4f}".format(loss_value), 
                                  'Res': "{0:.4f}".format(res_value), 
                                  'ICS': "{0:.4f}".format(ics__value),
                                  'Data': "{0:.4f}".format(data_value)})

    # Evaluates predictions at test points  
    # Evaluates predictions at test points  
    @partial(jit, static_argnums=(0,))
    def predict_full(self, params, U_star):
        s_pred =vmap(self.operator_net, (None, 0))(params, U_star)
        return s_pred


    
    def predict_res(self, params, U_star):
        res1, res2  =vmap(self.residual_net, (None, 0))(params, U_star)
        loss_res1 = jnp.mean((res1)**2, axis=1)
        loss_res2 = jnp.mean((res2)**2, axis=1)
        loss_res = loss_res1 + loss_res2
        return loss_res    
    