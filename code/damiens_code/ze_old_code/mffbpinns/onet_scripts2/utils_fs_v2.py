from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import math
from functools import wraps
import jax.numpy as jnp
from jax import random, jit, vmap
from jax.nn import swish
# import jax.lax as lax
import jax

from jax.flatten_util import ravel_pytree

from functools import partial
from torch.utils import data

# My Imports
from jax.tree_util import tree_map


def timing(f):
    """Decorator for measuring the execution time of methods."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        ts = time.time()
        result = f(*args, **kwargs)
        te = time.time()
        print("%r took %f s\n" % (f.__name__, te - ts))
        sys.stdout.flush()
        return result

    return wrapper


class DataGenerator(data.Dataset):
    def __init__(self, u, s, 
                 batch_size=64, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.u = u
        self.s = s
        
        self.N = u.shape[0]
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
        idx = random.choice(key, self.N, (self.batch_size,), replace=False)
        s = self.s[idx, :]
        u = self.u[idx, :]
        # Construct batch
        inputs = u
        outputs = s
        return inputs, outputs


class DataGenerator_res(data.Dataset):
    def __init__(self, u_coords,
                 batch_size=64, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.coords = u_coords
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
        u = self.coords[0] + (self.coords[1]-self.coords[0])*random.uniform(key, shape=[self.batch_size, 1])
        s = jnp.zeros([self.batch_size, 1])
        # Construct batch
        inputs = u
        outputs = s
        return inputs, outputs

class DataGenerator_res2(data.Dataset): # WORK IN PROGRESS
    def __init__(self, total_domain_bounds, model_prev, total_points, batch_size,
                 delta, Ndomains, step, k, c, key=random.PRNGKey(42)):
        
        # Make the domain arrays
        Tmax = total_domain_bounds[1] - total_domain_bounds[0]
        sigma = Tmax*delta/(2*(Ndomains - 1))
        mus = Tmax*jnp.linspace(0,1,Ndomains)
        double_domains = jnp.array([[mus[j+1] - sigma, mus[j] + sigma] for j in range(Ndomains - 1)])
        single_domains = jnp.array([[mus[j] + sigma, mus[j+2] - sigma] for j in range(Ndomains - 2)])
        if step == 0:
            single_domains = jnp.concatenate((jnp.array([[total_domain_bounds[0],double_domains[0][0]]]),
                                          jnp.array([[double_domains[-1][-1],total_domain_bounds[1]]])))
        else:
            single_domains = jnp.concatenate((jnp.array([[total_domain_bounds[0],double_domains[0][0]]]),
                                          single_domains,jnp.array([[double_domains[-1][-1],total_domain_bounds[1]]])))
        
        self.key = key
        self.delta = delta # overlap ratio
        self.Tmax = Tmax # max time to solve the pendulum
        self.Ndomains = Ndomains # number of domains on the current level of the domain decomposition (NOTE: This is Ndomains[-1] in the train_MF_EWC_script.py code)
        self.mus = mus # the centers of the NN domains (each domain is [mu - sigma, mu + sigma])
        self.sigma = sigma # half the length of the NN domain
        self.batch_size = batch_size
        self.total_domain_bounds = total_domain_bounds # domain bounds of the ENTIRE problem, [0, Tmax]
        self.single_domains = single_domains # domains where only one NN has support
        self.double_domains = double_domains # domains where two NNs have support
        self.single_domain_num_pts = self.point_allocation(single_domains,total_points)
        self.double_domain_num_pts = self.point_allocation(double_domains,total_points)
        self.single_domain_num_batch_pts = self.point_allocation(single_domains,batch_size)
        self.double_domain_num_batch_pts = self.point_allocation(double_domains,batch_size)
        self.total_points = jnp.sum(self.single_domain_num_pts) + jnp.sum(self.double_domain_num_pts)

        # get random points for single domains
        single_domain_key_array = random.split(key,len(self.single_domain_num_pts))
        self.key, subkey = random.split(self.key)
        single_domain_pts = []
        single_domain_probs = []
        params = model_prev.get_params(model_prev.opt_state)
        for idx in jnp.arange(len(single_domains)): # NOTE: this can probably be made more efficient with tree_map
            points = self.get_points(self.single_domains[idx],self.single_domain_num_pts[idx],single_domain_key_array[idx])
            res_val = model_prev.predict_res(params, points)
            err = res_val**k/jnp.mean(res_val**k) + c
            probabilities = err/jnp.sum(err)            
            single_domain_pts.append(points)
            single_domain_probs.append(probabilities)
        self.single_domain_pts = single_domain_pts
        self.single_domain_probs = single_domain_probs

        # get random points for double domains
        double_domain_key_array = random.split(self.key,len(self.double_domain_num_pts))
        self.key, subkey = random.split(self.key)
        double_domain_pts = []
        double_domain_weights = []
        double_domain_probs = []
        for idx in jnp.arange(len(double_domains)): # NOTE: this can probably be made more efficient with tree_map
            points = self.get_points(self.double_domains[idx],self.double_domain_num_pts[idx],double_domain_key_array[idx])
            weights = self.get_weights(self.mus[idx],self.mus[idx+1],self.sigma,points)
            res_val = model_prev.predict_res(params, points)
            err = res_val**k/jnp.mean(res_val**k) + c
            probabilities = err/jnp.sum(err)       
            double_domain_pts.append(points)
            double_domain_weights.append(weights)
            double_domain_probs.append(probabilities)
        self.double_domain_pts = double_domain_pts
        self.double_domain_weights = double_domain_weights
        self.double_domain_probs = double_domain_probs

    @partial(jit, static_argnums = (0,))
    def point_allocation(self,domains,N):
        """
        Determines how many points are to be allocated to each subdomain. NOTE: The number of total points
        assigned to all domains may be less than total_points due to rounding.
        ==================================================================================================
        INPUTS:
        domains:    A matrix where each row denotes the bounds of a domain.
        N:          Total number of points to be assigned
        OUTPUTS:
        temp:       A list of integers that determines how many points should be assigned to each domain
        """
        domain_lengths = (domains[:,1] - domains[:,0])/(self.total_domain_bounds[1] - self.total_domain_bounds[0]) 
        jnp.ravel(domain_lengths)
        domain_lengths = N*domain_lengths
        points = jnp.rint(domain_lengths).astype(jnp.int32)
        return points
    
    # @partial(jit, static_argnums = (0,))
    def get_points(self,domain,total_num_pts,key):
        """
        Generates uniformly random points on 'domain'
        """
        residual_pts = domain[0] + (domain[1] - domain[0])*random.uniform(key, shape=[total_num_pts, 1])
        return residual_pts
    
    @partial(jit, static_argnums = (0,))
    def get_weights(self,mu1,mu2,sigma,u):
        weights1 = (1 + jnp.cos(math.pi*(u-mu1)/sigma))**2
        weights2 = (1 + jnp.cos(math.pi*(u-mu2)/sigma))**2
        return {'left': weights1, 'right': weights2}

    def __getitem__(self, index):
        'Generate one batch of data'
        self.key, subkey = random.split(self.key)
        inputs = self.__data_generation(subkey)
        return inputs

    # @partial(jit, static_argnums=(0,))
    def __data_generation(self, key):
        single_domain_batches = []
        for idx in jnp.arange(len(self.single_domains)): # NOTE: this can probably be made more efficient with tree_map
            locs = random.choice(key, self.single_domain_num_pts[idx], (self.single_domain_num_batch_pts[idx],), p=self.single_domain_probs[idx], replace=False)
            batch_pts = self.single_domain_pts[idx][locs]
            single_domain_batches.append(batch_pts)

        double_domain_batches = []
        double_domain_left_weights = []
        double_domain_right_weights = []
        for idx in jnp.arange(len(self.double_domains)): # NOTE: this can probably be made more efficient with tree_map
            locs = random.choice(key, self.double_domain_num_pts[idx], (self.double_domain_num_batch_pts[idx],), p=self.double_domain_probs[idx], replace=False)
            batch_pts = self.double_domain_pts[idx][locs]
            batch_left_weights = self.double_domain_weights[idx]['left'][locs]
            batch_right_weights = self.double_domain_weights[idx]['right'][locs]
            double_domain_batches.append(batch_pts)
            double_domain_left_weights.append(batch_left_weights)
            double_domain_right_weights.append(batch_right_weights)


        return single_domain_batches, double_domain_batches, batch_left_weights, batch_right_weights














    
class DataGenerator_h(data.Dataset):
    def __init__(self, u, s,  u_l, 
                 batch_size=64, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.u = u
        self.s = s
        self.u_l = u_l
        
        self.N = u.shape[0]
        self.batch_size = batch_size
        self.key = rng_key

    def __getitem__(self, index):
        'Generate one batch of data'
        self.key, subkey = random.split(self.key)
        inputs, outputs,  u_l = self.__data_generation(subkey)
        return inputs, outputs,  u_l

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key):
        'Generates data containing batch_size samples'
        idx = random.choice(key, self.N, (self.batch_size,), replace=False)
        s = self.s[idx,:,:]
        u = self.u[idx,:,:]
        u_l = self.u_l[idx,:,:]

        # Construct batch
        inputs = (u, u_l)
        outputs = s
        return inputs, outputs
    

def DNN(branch_layers, activation=swish):

    def xavier_init_j(key, d_in, d_out):
        glorot_stddev = 1. / jnp.sqrt((d_in + d_out) / 2.)
        W = glorot_stddev * random.normal(key, (d_in, d_out))
        b = jnp.zeros(d_out)
        return W, b
    def init(rng_key1):
        def init_layer(key, d_in, d_out):
            k1, k2 = random.split(key)
            W, b = xavier_init_j(k1, d_in, d_out)
            return W, b
        key1, *keys1 = random.split(rng_key1, len(branch_layers))
        branch_params = list(map(init_layer, keys1, branch_layers[:-1], branch_layers[1:]))
        return branch_params
        
    def apply(params, u):
      #  print(u.shape)
        for k in range(len(branch_layers)-2):
            W_b, b_b = params[k]
            
            u = activation(jnp.dot(u, W_b) + b_b)

        W_b, b_b = params[-1]
        u = jnp.dot(u, W_b) + b_b
      #  print(u.shape)

        return u

    return init, apply



    
def nonlinear_DNN(branch_layers, activation=swish):

    def xavier_init_j(key, d_in, d_out):
        glorot_stddev = 1. / jnp.sqrt((d_in + d_out) / 2.)
        W = glorot_stddev * random.normal(key, (d_in, d_out))
        b = jnp.zeros(d_out)
        return W, b
    def init(rng_key1):
        def init_layer(key, d_in, d_out):
            k1, k2 = random.split(key)
            W, b = xavier_init_j(k1, d_in, d_out)
            return W, b
        key1, *keys1 = random.split(rng_key1, len(branch_layers))
        branch_params = list(map(init_layer, keys1, branch_layers[:-1], branch_layers[1:]))
        return branch_params
        
    def apply(params, u):
      #  print(u.shape)
        for k in range(len(branch_layers)-2):
            W_b, b_b = params[k]
            u = activation(jnp.dot(u, W_b) + b_b)
        W_b, b_b = params[-1]
        u = (jnp.dot(u, W_b) + b_b)
        return u
        
    def weight_norm(params):
    
        loss = 0

        for k in range(len(branch_layers)-1):
            W_b, b_b = params[k]
            
            loss += jnp.sum(W_b**2)
            loss += jnp.sum(b_b**2)

        return loss
    
    return init, apply, weight_norm

def linear_DNN(branch_layers):

    def xavier_init_j(key, d_in, d_out):
        glorot_stddev = 1. / jnp.sqrt((d_in + d_out) / 2.)
        W = glorot_stddev * random.normal(key, (d_in, d_out))
        b = jnp.zeros(d_out)
        return W, b
    def init(rng_key1):
        def init_layer(key, d_in, d_out):
            k1, k2 = random.split(key)
            W, b = xavier_init_j(k1, d_in, d_out)
            return W, b
        key1, *keys1 = random.split(rng_key1, len(branch_layers))
        branch_params = list(map(init_layer, keys1, branch_layers[:-1], branch_layers[1:]))
        return (branch_params )
        
    def apply(params, u):
        branch_params = params
        for k in range(len(branch_layers)-1):
            W_b, b_b = branch_params[k]

            u = (jnp.dot(u, W_b) + b_b)
        

        return u

    return init, apply


def linear_deeponet(branch_layers, trunk_layers):

    def xavier_init_j(key, d_in, d_out):
        glorot_stddev = 1. / jnp.sqrt((d_in + d_out) / 2.)
        W = glorot_stddev * random.normal(key, (d_in, d_out))
        b = jnp.zeros(d_out)
        return W, b
    def init(rng_key1, rng_key2):
        def init_layer(key, d_in, d_out):
            k1, k2 = random.split(key)
            W, b = xavier_init_j(k1, d_in, d_out)
            return W, b
        key1, *keys1 = random.split(rng_key1, len(branch_layers))
        key2, *keys2 = random.split(rng_key2, len(trunk_layers))
        branch_params = list(map(init_layer, keys1, branch_layers[:-1], branch_layers[1:]))
        trunk_params = list(map(init_layer, keys2, trunk_layers[:-1], trunk_layers[1:]))
        return (branch_params, trunk_params )
        
    def apply(params, u, y):
        branch_params, trunk_params = params
        for k in range(len(branch_layers)-1):
            W_b, b_b = branch_params[k]
            W_t, b_t = trunk_params[k]

            u = (jnp.dot(u, W_b) + b_b)
            y = (jnp.dot(y, W_t) + b_t)

        outputs = u*y

       # outputs = jnp.reshape(outputs,(outputs.shape[0], -1))
        
        return outputs

    return init, apply

###########################################################################################################
# This is an old weight function that I am not using anymore
###########################################################################################################

    # def weight_condition(self,condition,u,mu,sigma):
    #     w = lax.cond(condition, lambda u: (1 + jnp.cos(math.pi*(u-mu)/sigma))**2, lambda _: 0., u)
    #     return w
    
    # @partial(jit, static_argnums = (0,))
    # def w(self, mu, sigma, u):
    #     """
    #     Returns the weights of the points in "u"
    #     """
    #     conditions = (u < (mu + sigma)) & (u > (mu - sigma)) # NOTE: This condition should be able to be removed since all points lie in the support
    #     weights = vmap(self.weight_condition,(0,0,None,None))(conditions,u,mu,sigma)
    #     return weights
    

###########################################################################################################
# This is an old batching function where I was separately sampling uniform and residual points.
# I realized that sampling them separately is pointless. So, I am changing this.
###########################################################################################################

# class DataGenerator_res2(data.Dataset):
#     # def __init__(self, domain_bounds, single_domains, double_domains, N, key=random.PRNGKey(1234)):
#     def __init__(self, domain_bounds, model_prev, model_curr, single_domains, double_domains, 
#                  total_points, Tmax, delta, Ndomains, step, key=random.PRNGKey(42)):
        
#         # Make the domain arrays
#         sigma = Tmax*delta/(2*(Ndomains - 1))
#         mus = Tmax*jnp.linspace(0,1,Ndomains)
#         double_domains = jnp.array([[mus[j+1] - sigma, mus[j] + sigma] for j in range(Ndomains[-1] - 1)])
#         single_domains = jnp.array([[mus[j] + sigma, mus[j+2] - sigma] for j in range(Ndomains[-1] - 2)])
#         if step == 0:
#             single_domains = jnp.concatenate((jnp.array([[domain_bounds[0],double_domains[0][0]]]),
#                                           jnp.array([[double_domains[-1][-1],domain_bounds[1]]])))
#         else:
#             single_domains = jnp.concatenate((jnp.array([[domain_bounds[0],double_domains[0][0]]]),
#                                           single_domains,jnp.array([[double_domains[-1][-1],domain_bounds[1]]])))
        
#         self.delta = delta
#         self.Tmax = Tmax
#         self.Ndomains = Ndomains
#         self.mus = mus
#         self.sigma = sigma
#         self.domain_bounds = domain_bounds
#         self.single_domains = single_domains # domains where only one NN has support
#         self.double_domains = double_domains # domains where two NNs have support
#         self.single_domain_num_pts = self.point_allocation(single_domains,total_points)
#         self.double_domain_num_pts = self.point_allocation(double_domains,total_points)
#         self.total_points = jnp.sum(self.single_domain_num_pts) + jnp.sum(self.double_domain_num_pts)

#         # get random points for single domains
#         single_domain_key_array = random.split(key,len(self.single_domain_num_pts))
#         single_domain_pts = []
#         for idx in jnp.arange(len(single_domains)):
#             temp = self.get_points(self.single_domains[idx],self.single_domain_num_pts[idx],single_domain_key_array[idx])
#             single_domain_pts.append(temp)
#         self.single_domain_pts = single_domain_pts   

#         # get random points for double domains
#         double_domain_key_array = random.split(key,len(self.double_domain_num_pts))
#         double_domain_pts = []
#         for idx in jnp.arange(len(double_domains)):
#             temp = self.get_points(self.double_domains[idx],self.double_domain_num_pts[idx],double_domain_key_array[idx])
#             temp2 = self.get_weights()
#             double_domain_pts.append(temp)
#         self.double_domain_pts = double_domain_pts

#     def weight_condition(self,condition,u,mu,sigma):
#         w = lax.cond(condition, lambda u: (1 + jnp.cos(math.pi*(u-mu)/sigma))**2, lambda _: 0., u)
#         return w
    
#     # Changed this function
#     @partial(jit, static_argnums = (0,))
#     def w(self, mu, sigma, u):
#         conditions = (u < (mu + sigma)) & (u > (mu - sigma))
#         w_jl = vmap(self.weight_condition,(0,0,None,None))(conditions,u,mu,sigma)
#         return w_jl

#     @partial(jit, static_argnums = (0,))
#     def point_allocation(self,domains,N):
#         domain_lengths = (domains[:,1] - domains[:,0])/(self.domain_bounds[1] - self.domain_bounds[0]) 
#         jnp.ravel(domain_lengths)
#         domain_lengths = N*domain_lengths
#         temp = jnp.rint(domain_lengths).astype(jnp.int32)
#         return temp
    
#     def get_points(self,domain,total_num_pts,key):
#         key_res, key_uni = random.split(key)
#         residual_num_pts = total_num_pts // 2
#         uniform_num_pts = total_num_pts - residual_num_pts
#         residual_pts = domain[0] + (domain[1] - domain[0])*random.uniform(key_res, shape=[residual_num_pts, 1])
#         uniform_pts = domain[0] + (domain[1] - domain[0])*random.uniform(key_uni, shape=[uniform_num_pts, 1])
#         return {'res_pts': residual_pts, 'uni_pts': uniform_pts}
    
#     def __getitem__(self, index):
#         'Generate one batch of data'
#         self.key, key1, key2 = random.split(self.key,3)
#         inputs = self.__data_generation(key1,key2)
#         return inputs

#     @partial(jit, static_argnums=(0,))
#     def __data_generation(self, key1, key2):

#         pass