#============================================================================================
# My Imports
#============================================================================================

from anytree import NodeMixin
from MFDomainNet_Class import MFDomainNet
import numpy as onp

#============================================================================================
# Original Imports
#============================================================================================
import os

#import numpy as np
# import scipy.io
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import time
from utils_fs_v2 import timing,  DataGenerator, DNN
# import math
# import jax
import jax.numpy as np
from jax import random, grad, vmap, jit, hessian
from jax.example_libraries import optimizers
from jax.nn import relu, elu, log_softmax, softmax, swish
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
#import matplotlib.pyplot as plt
from copy import deepcopy

#============================================================================================
# Class Definitions
#============================================================================================

class RootUtilities:
    """
    Authors:        ADD
    Domain_Tree:    A super of the Domain class that stores important functions for working with
                    the domain decomposition tree.
    """
    def __init__(self):
        self.levels = [] # list of what classes are on which levels of the domain tree
        self.level_parameters = []
        self.level_vertices = []
        # self.predictions

class SFDomainNet(RootUtilities,NodeMixin):

    """
    Authors:            ADD
    Date:               July 3, 2023
    SFDomainNet:        Defines a single fidelity PINN (a classical PINN) on some nD rectangular domain. This
                        single fidelity network serves as the root of the domain network tree. Its' children
                        are multifidelity PINNS that use this network as a low fidelity approximation.
    =================================================================================================
    INPUT:
    sfnet_shape:        The layer structure of the low fidelity network (this is the same as the single 
                        fidelity network)
    ics_weight:         Penalty weight for not satisfying the initial conditions
    res_weight:         Penalty weight for the neural network not satisfying the physical constraints
    data_weight:        Penalty weight for the neural network not interpolating the given solution data
    params_prev:        List containing the parameters of a previously trained neural network
    lr:                 Determines the learning rate of the neural network.
    vertices:           Two opposite points on the hyperrectangle used to define the domain. 
    children:           The nodes whose domains are the immediate subsets of the current domain.
                        NOTE: Every child domain must be a subset of the parent domain
    """
    def __init__(self, sfnet_shape, ics_weight, res_weight, data_weight, pen_weight, params_prev, lr, vertices, children=None): 

        #===========================================================================================
        # My Initialization Code
        #===========================================================================================

        super(SFDomainNet,self).__init__()
        self.vertices = np.array(vertices) # two opposite vertices that define the n-dimensional box
        self.parent = None # parent domain of the current domain

        if children: # set children
            self.children = children
        
        # objects defined for the SFDomain.evaluate_neural_domain_tree(self,pts) function
        self.pts = None # the points on the interior of the root domain
        self.global_indices = None # NOTE: This is inefficient. Find a better way to set this

        #===========================================================================================
        # Original Initialization Code
        #===========================================================================================

        # Network initialization 
        self.init_sf, self.apply_sf = DNN(sfnet_shape)

        # If a non-empty list is passed in, load the variables in params_prev
        if len(params_prev) > 0:
            params = params_prev
        else:
            params_sf = self.init_sf(random.PRNGKey(1))
            params = (params_sf)

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
        self.data_weight = data_weight
        self.pen_weight = pen_weight

        # building loss function
        self.loss_training_log = []
        self.loss_res_log = []
        self.loss_ics_log = []
        self.loss_data_log = []

        self.tree_level_organizer(self) # add root to the level organizer

    # ===================================================================================================
    # Functions that I implemented
    # ===================================================================================================
    def tree_level_organizer(self,new_node): # TESTED
        """
        "tree_level_organizer" recognizes which level "new_node" is on and places "new_node"
        in the appropriate level in "self.levels".
        =================================================================================================
        INPUT:
        new_node:   A class that is being added to the tree
        NOTE: If the node that is being added has children, I do not believe it will add it to self.levels.
              This is something to work on maybe
        """
        depth = new_node.depth # find the depth of the new node
        params = self.get_params(new_node.opt_state) # NOTE: I am using the .get_params in SFDomainNet, not the MFDomainNet
        vertices = new_node.vertices
        if depth >= len(self.levels): # if "new_node" is the first on a new level, add that level to "self.levels"
            self.levels.append([new_node])
            self.level_parameters.append([params])
            self.level_vertices.append([vertices])
        else: # if "new_node" is a member of an existing level, add it to that level of "self.levels"
            self.levels[depth].append(new_node)
            self.level_parameters[depth].append(params)
            self.level_vertices[depth].append(vertices)

    def get_level_parameters(self,level):
        """
        "get_level_parameters" iterates through the domain nets contained in the level indicated by "level" 
        in "self.levels" and creates a list of the neural network parameters contained in the domain nets.
        =================================================================================================
        INPUT:
        level:          The level from which to retrieve the parameters of the domain nets
        OUTPUT:
        level_params:   List of the parameters of the neural networks on the current level.
        # NOTE: This function will probably be phased out because it is not necessary.
        """
        level_params = []
        for domain_net in self.levels[level]:
            params = domain_net.get_params(domain_net.opt_state)
            level_params.append(params)
        return level_params

    def find_interior_points(self,vertices,input_pts): # TESTED
        """
        "find_interior_points" determines which of the points in "input_pts" are on the interior of
        the hyperrectangle defined by the points in "vertices".
        =================================================================================================
        INPUT:
        vertices:   A list of two vertices that defines an nD hyperrectangle
        input_pts:  A list of nD points
        OUTPUT:
        indices:    The indices of the points in "input_pts" that are in the hyperrectangle
        pts:        The subset of points in "input_pts" that are in the hyperrectangle
        NOTE: This code can be vectorized. The for-loop will be slow.
        """
        indices = []
        pts = []
        number_of_points = len(input_pts)
        for i in range(number_of_points):
            condition = ((vertices[0] < input_pts[i]) & (input_pts[i] < vertices[1])).all()
            if condition:
                indices.append(i)
                pts.append(input_pts[i])
        return np.array(indices), np.array(pts)
    
    # ====================================================================================================
    # SINGLE FIDELITY: Don't mess with it for now.
    # ====================================================================================================
    
    def singlefidelity_network(self,params,pts):
        """
        "singlefidelity_network" evaluates the root network and outputs the solution at "pts".
        NOTE: This function is hard coded to solve the pendulum equation
        =================================================================================================
        INPUT:
        params:     The parameters to be passed into the neural network
        pts:        The points that the neural network is to be evaluated at.
        OUTPUT:
        s1,s2:      The position and velocity of the pendulum
        """
        B = self.apply_sf(params[0], pts)
        s1 = B[:1]
        s2 = B[1:]

        return s1, s2

    # Define ODE residual
    def residual_net(self, params, u, f):
        s1, s2 = f(params, u)

        def s1_fn(params, u):
          s1_fn, _ = f(params, u)
          return s1_fn[0]
        
        def s2_fn(params, u):
          _, s2_fn  = f(params, u)
          return s2_fn[0]

        s1_y = grad(s1_fn, argnums= 1)(params, u)
        s2_y = grad(s2_fn, argnums= 1)(params, u)

        res_1 = s1_y - s2
        res_2 = s2_y + 0.05 * s2 + 9.81 * np.sin(s1)

        return res_1, res_2
    
    def loss_data(self, params, batch,f):
        # Fetch data
        inputs, outputs = batch
        u = inputs
        
        s1 = outputs[:, 0:1]
        s2 = outputs[:, 1:2]

        # Compute forward pass
        s1_pred, s2_pred =vmap(f, (None, 0))(params, u)
        # Compute loss

        loss_s1 = np.mean((s1.flatten() - s1_pred.flatten())**2)
        loss_s2 = np.mean((s2.flatten() - s2_pred.flatten())**2)

        loss = loss_s1 + loss_s2
        return loss
    
    # Define residual loss
    def loss_res(self, params, batch,f):
        # Fetch data
        inputs, outputs = batch
        u = inputs

        # Compute forward pass
        res1_pred, res2_pred = vmap(self.residual_net, (None, 0, None))(params, u,f)
        # Compute loss

        loss_res1 = np.mean((res1_pred)**2)
        loss_res2 = np.mean((res2_pred)**2)
        loss_res = loss_res1 + loss_res2
        return loss_res
    
    # Define total loss
    def loss(self, params, ic_batch, res_batch, val_batch, level, f):
        loss_ics = self.loss_data(params, ic_batch, f)
        loss_res = self.loss_res(params, res_batch, f)
        loss_data = self.loss_data(params, val_batch, f)

        # NOTE: These lines penalize the network for having large network weights. I am removing this
        #       for now to simplify development. I will bring it back later. I also need to figure out how to
        #       make the weight penalization efficient for both the single fidelity and multifidelity networks
        #===============================================================================================
        # weights  = 0
        # for i in onp.arange(len(self.levels[level])):
        #      params_nl, params_l = params[i]
        #      weights += self.weight_nl(params_nl)

        # loss =  self.ics_weight*loss_ics + self.res_weight*loss_res +\
        #         self.data_weight*loss_data+ self.pen_weight*weights
        #===============================================================================================

        loss = self.ics_weight*loss_ics + self.res_weight*loss_res + self.data_weight*loss_data
        return loss 

    # Define a compiled update step
    @partial(jit, static_argnums=(0,5,6))
    def step(self, i, ic_batch, res_batch, val_batch, level, f):
        params = self.get_level_parameters(level)
        opt_state = self.opt_init(params) # NOTE: I have no idea if this is the correct way to do this
        # params = self.get_params(opt_state)

        g = grad(self.loss)(params, ic_batch, res_batch, val_batch, level, f)
        return self.opt_update(i, g, opt_state)

    # Optimize parameters in a loop
    # NOTE: For the single fidelity code, the "level" parameter must always be 0. Therefore,
    #       it is useless for the single fidelity case
    def train(self, ic_dataset, res_dataset, val_dataset, level, nIter = 10000):
        res_data = iter(res_dataset)
        ic_data = iter(ic_dataset)
        val_data = iter(val_dataset)

        if level == 0:
            training_function = self.singlefidelity_network
        else:
            training_function = self.multifidelity_network

        pbar = trange(nIter)
        # Main training loop
        for it in pbar:
            # Fetch data
            res_batch= next(res_data)
            ic_batch= next(ic_data)
            val_batch= next(val_data)

            self.opt_state = self.step(next(self.itercount), ic_batch, res_batch, val_batch, level, training_function)

            if it % 1000 == 0:
                params = self.get_params(self.opt_state)

                # Compute losses
                loss_value = self.loss(params, ic_batch, res_batch, val_batch, level, training_function)
                res_value = self.loss_res(params, res_batch, training_function)
                ics__value = self.loss_data(params, ic_batch, training_function)
                data_value = self.loss_data(params, val_batch, training_function)

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

    # @partial(jit,static_argnums=(0,2))
    def predict(self,pts,level):

        params = self.get_level_parameters(level)

        if level == 0:
            training_function = self.singlefidelity_network
        else:
            training_function = self.multifidelity_network

        u_pred = vmap(training_function, (None, 0))(params[0],pts)
        return u_pred
    
    # ====================================================================================================
    # MULTIFIDELITY CODE: Don't mess with things above this. I will probably end up
    # breaking them like last time.
    # ====================================================================================================
    
    # def multifidelity_network(self,params, params_prev,pts): # TESTED
    #     """
    #     "evaluate_neural_domain_tree" evaluates the neural domain tree layer by layer.
    #     =================================================================================================
    #     INPUT:
    #     pts:        A list of the collocation points that are to be evaluated by the neural domain tree
    #     OUTPUT:
    #     u_preds:    A matrix where the ith row corresponds to the prediction coming from the ith level
    #                 of the neural domain tree.
    #     NOTE: The current code only works if the support of the union of the domains at each level is the original domain.
    #     """
    #     # res = lax.cond(pts[0] > -10., lambda: 1., lambda: -1.)
    #     # self.pts = pts # save points to class
    #     # params_sf = self.get_params(self.opt_state)
    #     params_sf = params_prev[0][0] # fetch the parameters
    #     u_pred = self.apply_sf(params_sf,pts) # I added the [0] after the fact
    #     u_preds = np.zeros((len(self.levels),len(pts))) # create array to store predictions
    #     u_preds[0,:] = u_pred # the first prediction of the solution is the single fidelity net
    #     self.global_indices = np.arange(len(pts)) # NOTE: This is inefficient. Find a better way to set this
    #     iter = 1
    #     for level in self.levels[1:-1]: # iterates through the levels of the domain tree (skipping the first level which contains the root)
    #         u_pred = np.zeros(len(pts))
    #         for mfdomain in level: # iterates through the MFDomains in each level of the tree. NOTE: parallelize this loop
    #             parent = mfdomain.parent
    #             local_indices, mfdomain.pts = self.find_interior_points(mfdomain.vertices,parent.pts)
    #             mfdomain.global_indices = parent.global_indices[local_indices] # get the indices of the points in the support of mfdomain
    #             params_mf = mfdomain.get_params(mfdomain.opt_state)
    #             output = mfdomain.apply_mf(params_mf,mfdomain.pts,u_preds[-1][mfdomain.global_indices]) # analyze the local network
    #             u_pred[mfdomain.global_indices] += output # add the output to the prediction of the solution
    #         u_preds[iter,:] = u_pred # store the prediction on this level
    #         iter += 1
    #     return u_preds

    def multifidelity_network(self, params, params_prev, vertices, vertices_prev, level, pt):
        """
        NOTE: The current code only works if the support of the union of the domains at each level is the original domain.
        """
        params_sf = params_prev[0][0] # fetch the parameters for the single fidelity network
        u_pred = self.apply_sf(params_sf, pt)
        lax.fori_loop()


        return
    

    # Define ODE residual
    def residual_net_mf(self, params, params_prev, vertices, vertices_prev, level, u):
        s1, s2 = self.multifidelity_network(params, params_prev, vertices, vertices_prev, level, u)

        def s1_fn(params, u):
          s1_fn, _ = self.multifidelity_network(params, params_prev, vertices, vertices_prev, level, u)
          return s1_fn[0]
        
        def s2_fn(params, u):
          _, s2_fn  = self.multifidelity_network(params, params_prev, vertices, vertices_prev, level, u)
          return s2_fn[0]

        # NOTE: I may need to pass in all the parameters (params, params_prev, vertices, vertices_prev, level, u)
        #       into the below functions. I am not sure though.
        s1_y = grad(s1_fn, argnums= 2)(params, u)
        s2_y = grad(s2_fn, argnums= 2)(params, u)

        res_1 = s1_y - s2
        res_2 = s2_y + 0.05 * s2 + 9.81 * np.sin(s1)

        return res_1, res_2
    
    def loss_data_mf(self, params, params_prev, vertices, vertices_prev, level, batch):
        # Fetch data
        inputs, outputs = batch
        u = inputs
        
        s1 = outputs[:, 0:1]
        s2 = outputs[:, 1:2]

        # Compute forward pass
        s1_pred, s2_pred =vmap(self.multifidelity_network, (None, None, None, None, None, 0))(params, params_prev, vertices, vertices_prev, level, u)
        # Compute loss

        loss_s1 = np.mean((s1.flatten() - s1_pred.flatten())**2)
        loss_s2 = np.mean((s2.flatten() - s2_pred.flatten())**2)

        loss = loss_s1 + loss_s2
        return loss
    
    # Define residual loss
    def loss_res_mf(self, params, params_prev, vertices, vertices_prev, level, batch):
        # Fetch data
        inputs, outputs = batch
        u = inputs

        # Compute forward pass
        res1_pred, res2_pred = vmap(self.residual_net_mf, (None, None, None, None, None, 0))(params, params_prev, vertices, vertices_prev, level, u)
        # Compute loss

        loss_res1 = np.mean((res1_pred)**2)
        loss_res2 = np.mean((res2_pred)**2)
        loss_res = loss_res1 + loss_res2
        return loss_res
    
    # Define total loss
    def loss_mf(self, params, params_prev, vertices, vertices_prev, level, ic_batch, res_batch, val_batch):
        loss_res = self.loss_res_mf(params, params_prev, vertices, vertices_prev, level, res_batch)
        loss_ics = self.loss_data_mf(params, params_prev, vertices, vertices_prev, level, ic_batch)
        loss_data = self.loss_data_mf(params, params_prev, vertices, vertices_prev, level, val_batch)

        # NOTE: These lines penalize the network for having large network weights. I am removing this
        #       for now to simplify development. I will bring it back later. I also need to figure out how to
        #       make the weight penalization efficient for both the single fidelity and multifidelity networks
        #===============================================================================================
        # weights  = 0
        # for i in onp.arange(len(self.levels[level])):
        #      params_nl, params_l = params[i]
        #      weights += self.weight_nl(params_nl)

        # loss =  self.ics_weight*loss_ics + self.res_weight*loss_res +\
        #         self.data_weight*loss_data+ self.pen_weight*weights
        #===============================================================================================

        loss = self.ics_weight*loss_ics + self.res_weight*loss_res + self.data_weight*loss_data
        return loss 

    # Define a compiled update step
    @partial(jit, static_argnums=(0,))
    def step_mf(self, i, params_curr_lvl, params_prev_lvls, vertices, vertices_prev, level, ic_batch, res_batch, val_batch):
        # First index selects the network on the layer
        # Second index selects the linear vs nonlinear nets
        # Third index selects the layer in the network
        # Fourth index selects the matrix or shift vector

        # params = self.get_level_parameters(level)
        # opt_state = self.opt_init(params) # NOTE: I have no idea if this is the correct way to do this
        # params = self.get_params(opt_state)
        opt_state = self.opt_init(params_curr_lvl)

        g = grad(self.loss_mf)(params_curr_lvl, params_prev_lvls, vertices, vertices_prev, level, ic_batch, res_batch, val_batch)
        return self.opt_update(i, g, opt_state)

    # Optimize parameters in a loop
    # NOTE: I am using functions from "optimizers" that I do not fully understand. I am
     #      just hoping that they work out.
    def train_mf(self, ic_dataset, res_dataset, val_dataset, level, nIter = 10000):
        res_data = iter(res_dataset)
        ic_data = iter(ic_dataset)
        val_data = iter(val_dataset)

        params_prev_lvls = self.level_parameters[:level] # get parameters from the previous levels
        params_curr_lvl = self.level_parameters[level] # get the parameters of the current level
        vertices_prev = self.level_vertices[:level]
        vertices = self.level_vertices[level]

        pbar = trange(nIter)
        # Main training loop
        for it in pbar:
            # Fetch data
            res_batch= next(res_data)
            ic_batch= next(ic_data)
            val_batch= next(val_data)

            # self.opt_state = self.step_mf(next(self.itercount), ic_batch, res_batch, val_batch, params_curr_lvl, params_prev_lvls)
            opt_state = self.step_mf(next(self.itercount), params_curr_lvl, params_prev_lvls, vertices, vertices_prev, level, ic_batch, res_batch, val_batch)
            params_curr_lvl = self.get_params(opt_state)
            if it % 1000 == 0:
                params = self.get_params(opt_state)

                # Compute losses
                loss_value = self.loss_mf(params, ic_batch, res_batch, val_batch)
                res_value = self.loss_res_mf(params, res_batch)
                ics__value = self.loss_data_mf(params, ic_batch)
                data_value = self.loss_data_mf(params, val_batch)

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

    # @partial(jit,static_argnums=(0,2))
    def predict_mf(self,pts,level):

        params = self.get_level_parameters(level)

        if level == 0:
            training_function = self.singlefidelity_network
        else:
            training_function = self.multifidelity_network

        u_pred = vmap(training_function, (None, 0))(params[0],pts)
        return u_pred


    # ===================================================================================================
    # This is an ugly function that attempts to make the multifidelity network work. I was trying to
    # hack the batch in order to make things work. I am leaving this for now. I believe there is a better way.
    # ===================================================================================================


    # def multifidelity_network(self, params, pts): # NOT FINISHED
    #     self.pts = pts.val # points with the jax array
    #     empty_batch = pts.copy()
    #     empty_batch.val = np.array([]) # create an empty batch to add points to later
    #     params_sf = self.get_params(self.opt_state)
    #     u_sf = self.apply_sf(params_sf,pts) # single fidelity prediction
    #     u_preds = [u_sf.val] # I am not pre-defining the array this time. I am going to simply append the new solution to the previous one.
    #     self.global_indices = np.arange(len(pts)) # NOTE: I am not sure that this works. Even if it does, it is inefficient. Find a better way to set this.
    #     iter = 1
    #     for level in self.levels[1:-1]: # iterates through the levels of the domain tree (skipping the first and last levels)
    #         u_pred = np.zeros(len(pts.val))
    #         for mfdomain in level: # iterates through the MFDomains in each level of the tree. NOTE: parallelize this loop
    #             parent = mfdomain.parent
    #             local_indices, mfdomain.pts = self.find_interior_points(mfdomain.vertices,parent.pts)
    #             mfdomain.global_indices = parent.global_indices[local_indices] # get the indices of the points in the support of mfdomain
    #             current_params = mfdomain.get_params(mfdomain.opt_state)
    #             batch_pts = empty_batch.copy()
    #             batch_pts.val = mfdomain.pts # make a batch with mfdomain.pts
    #             batch_u_preds = empty_batch.copy()
    #             batch_u_preds.val = u_preds[-1][mfdomain.global_indices] # make a batch with u_preds[-1][mfdomain.global_indices]
    #             output = mfdomain.apply_mf(current_params,batch_pts,batch_u_preds) # analyze the local network
    #             u_pred[mfdomain.global_indices] += output.val # add the output to the prediction of the solution
    #         u_preds.append(u_pred)
    #         iter += 1

    #     # NEED TO ADD CODE FOR EVALUATING THE FINAL LAYER
    #     return u_preds


    # ===================================================================================================
    # Functions that were already implemented in MF_EWC_Class.py
    # ===================================================================================================

    # def operator_net(self, params, u):
    
    #     ul = self.apply_lf(self.params_A, u)
    
    #     for j in onp.arange(len(self.Ndomains)-1):
    #         ul_cur = 0
    #         for i in onp.arange(self.Ndomains[j]):
                
    #             paramsB_nl =  self.params_t[j][i][0]
    #             paramsB_l =  self.params_t[j][i][1]
    #             y = np.hstack([u, ul])

    #             u_nl = self.apply_nl(paramsB_nl, y)
    #             u_l = self.apply_l(paramsB_l, ul)
                
    #             w = self.w_jl(i, self.Ndomains[j], u)
    #             ul_cur += w*(u_l + u_nl)

    #         ul = ul_cur
        
    #     s1 = 0
    #     s2 = 0
    #     for i in onp.arange(self.Ndomains[-1]):
    #         params_nl, params_l = params[i]
    #         y = np.hstack([u, ul])

    #         u_nl = self.apply_nl(params_nl, y)
    #         u_l = self.apply_l(params_l, ul)
            
    #         w = self.w_jl(i, self.Ndomains[-1], u)
    #         s1 += w*(u_l[:1]+ u_nl[:1])
    #         s2 += w*(u_l[1:]+ u_nl[1:])
        
    #     return s1, s2
    

    # # Define ODE residual
    # def residual_net(self, params, u):

    #     s1, s2 = self.operator_net(params, u)
    #  #   print(s1.shape)
    #  #   print(u.shape)


    #     def s1_fn(params, u):
    #       s1_fn, _ = self.operator_net(params, u)
    #    #   print(s1_fn.shape)
    #       return s1_fn[0]
        
    #     def s2_fn(params, u):
    #       _, s2_fn  = self.operator_net(params, u)
    #       return s2_fn[0]

    #     s1_y = grad(s1_fn, argnums= 1)(params, u)
    #     s2_y = grad(s2_fn, argnums= 1)(params, u)

    #     res_1 = s1_y - s2
    #     res_2 = s2_y + 0.05 * s2 + 9.81 * np.sin(s1)

    #     return res_1, res_2


    # def loss_data(self, params, batch):
    #     # Fetch data
    #     inputs, outputs = batch
    #     u = inputs
        
    #     s1 = outputs[:, 0:1]
    #     s2 = outputs[:, 1:2]

    #     # Compute forward pass
    #     s1_pred, s2_pred =vmap(self.operator_net, (None, 0))(params, u)
    #     # Compute loss

    #     loss_s1 = np.mean((s1.flatten() - s1_pred.flatten())**2)
    #     loss_s2 = np.mean((s2.flatten() - s2_pred.flatten())**2)

    #     loss = loss_s1 + loss_s2
    #     return loss
    
    # # Define residual loss
    # def loss_res(self, params, batch):
    #     # Fetch data
    #     inputs, outputs = batch
    #     u = inputs

    #     # Compute forward pass
    #     res1_pred, res2_pred = vmap(self.residual_net, (None, 0))(params, u)
    #     # Compute loss

    #     loss_res1 = np.mean((res1_pred)**2)
    #     loss_res2 = np.mean((res2_pred)**2)
    #     loss_res = loss_res1 + loss_res2
    #     return loss_res

    # # Define total loss
    # def loss(self, params, ic_batch, res_batch, val_batch):
    #     loss_ics = self.loss_data(params, ic_batch)
    #     loss_res = self.loss_res(params, res_batch)
    #     loss_data = self.loss_data(params, val_batch)
        
    #     weights  = 0
    #     for i in onp.arange(self.Ndomains[-1]):
    #          params_nl, params_l = params[i]
    #          weights += self.weight_nl(params_nl)

    #     loss =  self.ics_weight*loss_ics + self.res_weight*loss_res +\
    #         self.data_weight*loss_data+ self.pen_weight*weights
            
    #     return loss 
    

    
    #     # Define a compiled update step
    # @partial(jit, static_argnums=(0,))
    # def step(self, i, opt_state, ic_batch, res_batch, val_batch):
    #     params = self.get_params(opt_state)

    #     g = grad(self.loss)(params, ic_batch, res_batch, val_batch)
    #     return self.opt_update(i, g, opt_state)
    

    # # Optimize parameters in a loop
    # def train(self, ic_dataset, res_dataset, val_dataset, nIter = 10000):
    #     res_data = iter(res_dataset)
    #     ic_data = iter(ic_dataset)
    #     val_data = iter(val_dataset)

    #     pbar = trange(nIter)
    #     # Main training loop
    #     for it in pbar:
    #         # Fetch data
    #         res_batch= next(res_data)
    #         ic_batch= next(ic_data)
    #         val_batch= next(val_data)

    #         self.opt_state = self.step(next(self.itercount), self.opt_state, 
    #                                    ic_batch, res_batch, val_batch)
            
    #         if it % 1000 == 0:
    #             params = self.get_params(self.opt_state)

    #             # Compute losses
    #             loss_value = self.loss(params, ic_batch, res_batch, val_batch)
    #             res_value = self.loss_res(params, res_batch)
    #             ics__value = self.loss_data(params, ic_batch)
    #             data_value = self.loss_data(params, val_batch)

    #             # Store losses
    #             self.loss_training_log.append(loss_value)
    #             self.loss_res_log.append(res_value)
    #             self.loss_ics_log.append(ics__value)
    #             self.loss_data_log.append(data_value)

    #             # Print losses
    #             pbar.set_postfix({'Loss': "{0:.4f}".format(loss_value), 
    #                               'Res': "{0:.4f}".format(res_value), 
    #                               'ICS': "{0:.4f}".format(ics__value),
    #                               'Data': "{0:.4f}".format(data_value)})

    # # Evaluates predictions at test points  
    # # Evaluates predictions at test points  
    # @partial(jit, static_argnums=(0,))
    # def predict_full(self, params, U_star):
    #     s_pred =vmap(self.operator_net, (None, 0))(params, U_star)
    #     return s_pred

    # def predict_res(self, params, U_star):
    #     res1, res2  =vmap(self.residual_net, (None, 0))(params, U_star)
    #     loss_res1 = np.mean((res1)**2, axis=1)
    #     loss_res2 = np.mean((res2)**2, axis=1)
    #     loss_res = loss_res1 + loss_res2
    #     return loss_res    

#==========================================================================================================
# Tested code for seeing how to evaluate the network layer by layer
#==========================================================================================================

    # def evaluate_neural_domain_tree(self,pts): # TESTED
    #     """
    #     "evaluate_neural_domain_tree" evaluates the neural domain tree layer by layer.
    #     =================================================================================================
    #     INPUT:
    #     pts:        A list of the collocation points that are to be evaluated by the neural domain tree
    #     OUTPUT:
    #     u_preds:    A matrix where the ith row corresponds to the prediction coming from the ith level
    #                 of the neural domain tree.
    #     NOTE: The current code only works if the support of the union of the domains at each level is the original domain.
    #     """
    #     self.pts = pts # save points to class
    #     params = self.get_params(self.opt_state)
    #     u_pred = self.apply_sf(params,pts)
    #     u_preds = np.zeros((len(self.levels),len(pts))) # create array to store predictions
    #     u_preds[0,:] = u_pred # the first prediction of the solution is the single fidelity net
    #     self.global_indices = np.arange(len(pts)) # NOTE: This is inefficient. Find a better way to set this
    #     iter = 1
    #     for level in self.levels[1:]: # iterates through the levels of the domain tree (skipping the first level which contains the root)
    #         u_pred = np.zeros(len(pts))
    #         for mfdomain in level: # iterates through the MFDomains in each level of the tree. NOTE: parallelize this loop
    #             parent = mfdomain.parent
    #             local_indices, mfdomain.pts = self.find_interior_points(mfdomain.vertices,parent.pts)
    #             mfdomain.global_indices = parent.global_indices[local_indices] # get the indices of the points in the support of mfdomain
    #             params = mfdomain.get_params(mfdomain.opt_state)
    #             output = mfdomain.apply_mf(params,mfdomain.pts,u_preds[-1][mfdomain.global_indices]) # analyze the local network
    #             u_pred[mfdomain.global_indices] += output # add the output to the prediction of the solution
    #         u_preds[iter,:] = u_pred # store the prediction on this level
    #         iter += 1
    #     return u_preds

    # def evaluate_neural_domain_tree(self,pts): # TESTED
    #     """
    #     "evaluate_neural_domain_tree" evaluates the neural domain tree layer by layer.
    #     =================================================================================================
    #     INPUT:
    #     pts:        A list of the collocation points that are to be evaluated by the neural domain tree
    #     OUTPUT:
    #     u_preds:    A matrix where the ith row corresponds to the prediction coming from the ith level
    #                 of the neural domain tree.
    #     NOTE: The current code only works if the support of the union of the domains at each level is the original domain.
    #     """
    #     self.pts = pts # save points to class
    #     params = self.get_params(self.opt_state)
    #     # u_pred = self.apply_sf(params,pts)
    #     u_pred = vmap(self.apply_sf,(None,0))(params,pts)
    #     u_preds = np.zeros((len(self.levels),len(pts))) # create array to store predictions
    #     # u_preds[0,:] = u_pred # the first prediction of the solution is the single fidelity net
    #     u_preds.at[0,:].set(u_pred)
    #     self.global_indices = np.arange(len(pts)) # NOTE: This is inefficient. Find a better way to set this
    #     iter = 1
    #     for level in self.levels[1:]: # iterates through the levels of the domain tree (skipping the first level which contains the root)
    #         u_pred = np.zeros(len(pts))
    #         for mfdomain in level: # iterates through the MFDomains in each level of the tree. NOTE: parallelize this loop
    #             parent = mfdomain.parent     
    #             local_indices, mfdomain.pts = self.find_interior_points(mfdomain.vertices,parent.pts)
    #             mfdomain.global_indices = parent.global_indices[local_indices] # get the indices of the points in the support of mfdomain
    #             params = mfdomain.get_params(mfdomain.opt_state)
    #             output = vmap(mfdomain.apply_mf,(None,0,0))(params,mfdomain.pts,u_preds[-1][mfdomain.global_indices]) # analyze the local network
    #             # output = mfdomain.apply_mf(params,mfdomain.pts,u_preds[-1][mfdomain.global_indices]) # analyze the local network
    #             u_pred[mfdomain.global_indices] += output # add the output to the prediction of the solution
    #         u_preds[iter,:] = u_pred # store the prediction on this level
    #         iter += 1
    #     return u_preds


#==========================================================================================================
# A method for finding the interior points that I am not using
#==========================================================================================================
                    

    # def find_interior_points(self,verts,parent_pts):
    #     """
    #     Creates a mask that communicates the locations of which points in "pts" are in the hyperrectangle
    #     defined by verts.
    #     """
    #     indices = []
    #     pts = []
    #     number_of_points = len(parent_pts)
    #     for i in range(number_of_points):
    #         condition = ((verts[0] <= parent_pts[i]) & (parent_pts[i] <= verts[1])).all()
    #         if condition:
    #             indices.append(i)
    #             pts.append(parent_pts[i])
    #     return indices

#==========================================================================================================
# I coded this up for evaluation, only to realize the idea here is wrong. I need to evaluate all the 
# neural networks on each layer before progressing to the next layer. In this code I proceed like a DFS.
# So, the low-fidelity approximation on the previous layer is not complete.
#==========================================================================================================

    # def find_interior_pts(self,verts,pts):
    #     """
    #     Creates a mask that communicates the locations of which points in "pts" are in the hyperrectangle
    #     defined by verts.
    #     """
    #     mask = [((verts[0] <= pt) & (pt <= verts[1])).all() for pt in pts] # find which of the parent's points are in the current domain
    #     return mask

    # def evaluate_neural_domain_tree(self,pts):
    #     u_curr = self.apply_sf(self.params,pts)
    #     masks = []
    #     u_locals = []
    #     for child in self.children:
    #         mask = self.find_interior_pts(child.vertices,pts)
    #         masks.append(mask)
    #         u_local = self.recursive_evaluator(child,u_curr[mask],pts[mask])
    #         u_locals.append(u_local)

    # def recursive_evaluator(self,domain,u_lf,pts):
    #     u_curr = domain.apply_mf(domain.params,u_lf,pts)
    #     if domain.is_leaf:
    #         u_global = u_curr
    #     else:
    #         masks = []
    #         u_locals = []
    #         for child in domain.children:
    #             mask = self.find_interior_pts(child.vertices,pts)
    #             masks.append(mask)
    #             u_local = self.recursive_evaluator(child,u_curr[mask],pts[mask])
    #             u_locals.append(u_local)
    #         u_global = np.zeros(len(pts))
    #         for i in len(masks):
    #             temp = np.zeros(len(pts))
    #             mask = masks[i]
    #             u_local = u_locals[i]
    #             inc = 0
    #             for j in len(pts):
    #                 if mask[j]:
    #                     temp[j] = u_local[inc]
    #                     inc = inc + 1
    #                 else:
    #                     pass
    #             u_global = u_global + temp
    #     return u_global
        



#==========================================================================================================
# Attempt at adaptively adding new domains to the tree
#==========================================================================================================

    # def add_domains(self,vertices_list):
    #     # this function adds a level to the neural domain tree
    #     if self.depth == 0: # if the network depth is only the root make the root the parent of all the vertices
    #         for vertices in vertices_list:
    #             MFDomainNet(vertices,parent=self)
    #     else:
    #         parents = self.find_parents(vertices_list)
    #         for parent in parents:
    #             MFDomainNet(vertices,parent=parent)

    # def find_parent(self,vertices):
    #     if self.is_leaf == True:
    #         return self
    #     parent = self
    #     flag = True
    #     while flag:
    #         flag = False
    #         for child in parent.children:
    #             condition = ((child.vertices[0] <= vertices[0]) & (vertices[1] <= child.vertices[1])).all()
    #             if(condition):
    #                 flag = True
    #                 parent = child
    #                 break

    #     return parent

    # def find_parents(self,vertices_list):
    #     parents = []
    #     for vertices in vertices_list:
    #         parent = self.find_parent(vertices)
    #         parents.append(parent)

    


    