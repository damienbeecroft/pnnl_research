#============================================================================================
# My Imports
#============================================================================================

from anytree import NodeMixin
from MFDomainNet_Class import MFDomainNet

#============================================================================================
# Original Imports
#============================================================================================
import os

#import numpy as np
import scipy.io
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import time
from utils_fs_v2 import timing,  DataGenerator, DNN
import math
import jax
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
        # self.predictions

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
        if depth >= len(self.levels): # if "new_node" is the first on a new level, add that level to "self.levels"
            self.levels.append([new_node])
        else: # if "new_node" is a member of an existing level, add it to that level of "self.levels"
            self.levels[depth].append(new_node)

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
    def __init__(self, sfnet_shape, ics_weight, res_weight, data_weight, params_prev, lr, vertices, children=None): 

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

        # building loss function
        self.loss_training_log = []
        self.loss_res_log = []
        self.loss_ics_log = []
        self.loss_data_log = []

        self.tree_level_organizer(self) # add root to the level organizer

    # =============================================
    # Evaluation
    # =============================================.

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

    def evaluate_neural_domain_tree(self,pts): # TESTED
        """
        "evaluate_neural_domain_tree" evaluates the neural domain tree layer by layer.
        =================================================================================================
        INPUT:
        pts:        A list of the collocation points that are to be evaluated by the neural domain tree
        OUTPUT:
        u_preds:    A matrix where the ith row corresponds to the prediction coming from the ith level
                    of the neural domain tree.
        NOTE: The current code only works if the support of the union of the domains at each level is the original domain.
        """
        self.pts = pts # save points to class
        params = self.get_params(self.opt_state)
        # u_pred = self.apply_sf(params,pts)
        u_pred = vmap(self.apply_sf,(None,0))(params,pts)
        u_preds = np.zeros((len(self.levels),len(pts))) # create array to store predictions
        u_preds[0,:] = u_pred # the first prediction of the solution is the single fidelity net
        self.global_indices = np.arange(len(pts)) # NOTE: This is inefficient. Find a better way to set this
        iter = 1
        for level in self.levels[1:]: # iterates through the levels of the domain tree (skipping the first level which contains the root)
            u_pred = np.zeros(len(pts))
            for mfdomain in level: # iterates through the MFDomains in each level of the tree. NOTE: parallelize this loop
                parent = mfdomain.parent     
                local_indices, mfdomain.pts = self.find_interior_points(mfdomain.vertices,parent.pts)
                mfdomain.global_indices = parent.global_indices[local_indices] # get the indices of the points in the support of mfdomain
                params = mfdomain.get_params(mfdomain.opt_state)
                output = vmap(mfdomain.apply_mf,(None,0,0))(params,mfdomain.pts,u_preds[-1][mfdomain.global_indices])
                # output = mfdomain.apply_mf(params,mfdomain.pts,u_preds[-1][mfdomain.global_indices]) # analyze the local network
                u_pred[mfdomain.global_indices] += output # add the output to the prediction of the solution
            u_preds[iter,:] = u_pred # store the prediction on this level
            iter += 1
        return u_preds























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

    


    