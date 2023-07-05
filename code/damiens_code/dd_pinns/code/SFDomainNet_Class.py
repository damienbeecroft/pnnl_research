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
        # self.depth = 1 # initial depth of the network is one
        # self.dnn_size = [3,80,80,80,2]
        # self.nonlin_size = [3,80,80,80,2]
        # self.lin_size = [3,4,2]
        pass
        


class SFDomainNet(RootUtilities,NodeMixin):

    """
    Authors:            ADD
    Date:               July 3, 2023
    SFDomainNet:        Defines a single fidelity PINN (a classical PINN) on some nD rectangular domain. This
                        single fidelity network serves as the root of the domain network tree. Its' children
                        are multifidelity PINNS that use this network as a low fidelity approximation.
    =================================================================================================
    *args:
    layers_branch_low:  The layer structure of the low fidelity network (this is the same as the single 
                        fidelity network)
    ics_weight:         Penalty weight for not satisfying the initial conditions
    res_weight:         Penalty weight for the neural network not satisfying the physical constraints
    data_weight:        Penalty weight for the neural network not interpolating the given solution data
    params_prev:        List containing the parameters of a previously trained neural network
    lr:                 Determines the learning rate of the neural network.
    vertices:           Two opposite points on the hyperrectangle used to define the domain. 
    **kwargs:
    parent:             The node corresponding to the domain that the current node's domain is the immediate subset of
    children:           The nodes whose domains are the immediate subsets of the current domain.
    """
    def __init__(self, sfnet_shape, nonlin_mfnet_shape, lin_mfnet_shape, ics_weight, res_weight, data_weight, params_prev, lr, 
                 vertices, children=None): 

        #===========================================================================================
        # My Initialization Code
        #===========================================================================================

        super(SFDomainNet,self).__init__()
        self.vertices = vertices # two opposite vertices that define the n-dimensional box
        self.parent = None # parent domain of the current domain
        self.lvl = 0

        if children: # set children
            self.children = children
            # self.depth +=1 # increase network depth by one

        #===========================================================================================
        # Original Initialization Code
        #===========================================================================================

        # Network initialization 
        self.init_low, self.apply_low = DNN(sfnet_shape)
        params_low = self.init_low(random.PRNGKey(1))
        params = (params_low)

        # If a non-empty list is passed in, load the variables in params_prev
        if len(params_prev) > 0:
             params = params_prev
        
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

    # =============================================
    # evaluation
    # =============================================

    def add_domains(self,vertices_list):
        # this function adds a level to the neural domain tree
        if self.depth == 0: # if the network depth is only the root make the root the parent of all the vertices
            for vertices in vertices_list:
                MFDomainNet(vertices,parent=self)
        else:
            parents = self.find_parents(vertices_list)
            for parent in parents:
                MFDomainNet(vertices,parent=parent)

    def find_parent(self,vertices):
        if self.is_leaf == True:
            return self
        parent = self
        flag = True
        while flag:
            flag = False
            for child in parent.children:
                condition = ((child.vertices[0] <= vertices[0]) & (vertices[1] <= child.vertices[1])).all()
                if(condition):
                    flag = True
                    parent = child
                    break

        return parent

    def find_parents(self,vertices_list):
        parents = []
        for vertices in vertices_list:
            parent = self.find_parent(vertices)
            parents.append(parent)

    # def is_subdomain(self,vertices1,vertices2):

            


    # def predict_solution(self):
    #     pass
    


    