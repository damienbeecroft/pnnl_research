#============================================================================================
# My Imports
#============================================================================================

from anytree import NodeMixin

#============================================================================================
# Original Imports
#============================================================================================

import os

#import numpy as np
# import scipy.io
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# import time
from utils_fs_v2 import timing,DataGenerator, DataGenerator_h, nonlinear_DNN, linear_DNN, DNN, linear_deeponet
import math
# import jax
import jax.numpy as np
from jax import random, grad, vmap, jit, hessian
from jax.example_libraries import optimizers
# from jax.experimental.ode import odeint
# from jax.nn import relu, elu, log_softmax, softmax
# from jax.config import config
#from jax.ops import index_update, index
# from jax import lax
from jax.flatten_util import ravel_pytree

import itertools
# from functools import partial
# from torch.utils import data
# from tqdm import trange, tqdm
#import matplotlib.pyplot as plt
#import pandas as pd
#import matplotlib.pyplot as plt
# from copy import deepcopy
# import numpy as onp

#============================================================================================
# Class Definitions
#============================================================================================

class MFDomainNet(NodeMixin):
    # nonlin_mfnet_shape, lin_mfnet_shape, 
    
    def __init__(self, nonlin_mfnet_shape, lin_mfnet_shape, ics_weight, res_weight, data_weight, 
                 pen_weight, lr, params_prev, vertices, parent = None, children=None):

        #===========================================================================================
        # My Initialization Code
        #===========================================================================================

        super(MFDomainNet,self).__init__()
        self.vertices = np.array(vertices) # two opposite vertices that define the n-dimensional box
        self.parent = parent # parent domain of the current domain

        if children: # set children
            self.children = children

        # this is used to record which points in SFDomain.evaluate_neural_domain_tree(self,pts) are
        # on the interior of the current node's domain          
        self.global_indices = []
        self.pts = []

        #===========================================================================================
        # Original Initialization Code
        #===========================================================================================

        self.init_nl, self.apply_nl, self.weight_nl = nonlinear_DNN(nonlin_mfnet_shape)
        self.init_l, self.apply_l = linear_DNN(lin_mfnet_shape)

        if len(params_prev) > 0:
            params = params_prev
        else:
            params_nl = self.init_nl(random.PRNGKey(13))
            params_l = self.init_l(random.PRNGKey(12345))
            params = (params_l, params_nl)

        # NOTE: The below definitions between the equals signs are likely not necessary and simply take up space. 
        #       These things should be added to the SFDomainNet class
        #===========================================================================================
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
        #===========================================================================================

        self.root.tree_level_organizer(self) # add current node to the level organizer 

    # =============================================
    # Evaluation
    # =============================================

    def weight(self,x):
        """
        NOTE: In Alexander's paper, this is w hat, not w. Amanda does not ensure that the weight
              functions constitute a partition of unity globally. I am not sure if this is important?
        """
        vertices = self.vertices
        mu = (vertices[0] + vertices[1])/2
        sigma = (vertices[1] - vertices[0])/2
        w = 1 + np.cos(math.pi*(x-mu)/sigma)
        w = w**2
        return w

    def apply_mf(self,params,pts,u_lf):
        # params = self.unravel_params # NOTE: This is probably wrong. I am not certain what format the two functions below are expecting
        u_nl = self.apply_nl(params[1], np.hstack([pts, u_lf]))
        u_l = self.apply_l(params[0], u_lf)
        w = self.weight(pts)
        u_local = w*(u_l + u_nl)
        return u_local

