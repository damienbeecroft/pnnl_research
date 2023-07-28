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
import jax.lax as lax

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
    
    # This is the original code for the residual data set generator
    # res_pts = coords[0] + (coords[1]-coords[0])*random.uniform(key, shape=[20000,1])
    # res_val = model_A.predict_res(params_A, res_pts)
    # err = res_val**k/np.mean( res_val**k) + c
    # err_norm = err/np.sum(err)                        
    # res_dataset = DataGenerator_res2(coords, res_pts, err_norm, batch_size_res, batch_size)

class DataGenerator_res2(data.Dataset):
    # def __init__(self, domain_bounds, single_domains, double_domains, N, key=random.PRNGKey(1234)):
    def __init__(self, domain_bounds, model_prev, model_curr, single_domains, double_domains, 
                 total_points, Tmax, delta, Ndomains, step, key=random.PRNGKey(42)):
        
        # Make the domain arrays
        sigma = Tmax*delta/(2*(Ndomains - 1))
        mus = Tmax*jnp.linspace(0,1,Ndomains)
        double_domains = jnp.array([[mus[j+1] - sigma, mus[j] + sigma] for j in range(Ndomains[-1] - 1)])
        single_domains = jnp.array([[mus[j] + sigma, mus[j+2] - sigma] for j in range(Ndomains[-1] - 2)])
        if step == 0:
            single_domains = jnp.concatenate((jnp.array([[domain_bounds[0],double_domains[0][0]]]),
                                          jnp.array([[double_domains[-1][-1],domain_bounds[1]]])))
        else:
            single_domains = jnp.concatenate((jnp.array([[domain_bounds[0],double_domains[0][0]]]),
                                          single_domains,jnp.array([[double_domains[-1][-1],domain_bounds[1]]])))
        
        self.delta = delta
        self.Tmax = Tmax
        self.Ndomains = Ndomains
        self.mus = mus
        self.sigma = sigma
        self.domain_bounds = domain_bounds
        self.single_domains = single_domains # domains where only one NN has support
        self.double_domains = double_domains # domains where two NNs have support
        self.single_local_points = self.point_allocation(single_domains,total_points)
        self.double_local_points = self.point_allocation(double_domains,total_points)
        self.total_points = jnp.sum(self.single_local_points) + jnp.sum(self.double_local_points)

        # get random points for single domains
        single_domain_key_array = random.split(key,len(self.single_local_points))
        single_domain_point_dict = []
        for idx in jnp.arange(len(single_domains)):
            temp = self.get_points(self.single_domains[idx],self.single_local_points[idx],single_domain_key_array[idx])
            single_domain_point_dict.append(temp)
        self.single_domain_point_dict = single_domain_point_dict   

        # get random points for double domains
        double_domain_key_array = random.split(key,len(self.double_local_points))
        double_domain_point_dict = []
        for idx in jnp.arange(len(double_domains)):
            temp = self.get_points(self.double_domains[idx],self.double_local_points[idx],double_domain_key_array[idx])
            temp2 = self.get_weights()
            double_domain_point_dict.append(temp)
        self.double_domain_point_dict = double_domain_point_dict

    def weight_condition(self,condition,u,mu,sigma):
        w = lax.cond(condition, lambda u: (1 + jnp.cos(math.pi*(u-mu)/sigma))**2, lambda _: 0., u)
        return w
    
    # Changed this function
    @partial(jit, static_argnums = (0,))
    def w(self, mu, sigma, u):
        conditions = (u < (mu + sigma)) & (u > (mu - sigma))
        w_jl = vmap(self.weight_condition,(0,0,None,None))(conditions,u,mu,sigma)
        return w_jl

    @partial(jit, static_argnums = (0,))
    def point_allocation(self,domains,N):
        domain_lengths = (domains[:,1] - domains[:,0])/(self.domain_bounds[1] - self.domain_bounds[0]) 
        jnp.ravel(domain_lengths)
        domain_lengths = N*domain_lengths
        temp = jnp.rint(domain_lengths).astype(jnp.int32)
        return temp
    
    def get_points(self,domain,total_num_pts,key):
        key_res, key_uni = random.split(key)
        residual_num_pts = total_num_pts // 2
        uniform_num_pts = total_num_pts - residual_num_pts
        residual_pts = domain[0] + (domain[1] - domain[0])*random.uniform(key_res, shape=[residual_num_pts, 1])
        uniform_pts = domain[0] + (domain[1] - domain[0])*random.uniform(key_uni, shape=[uniform_num_pts, 1])
        return {'res_pts': residual_pts, 'uni_pts': uniform_pts}
    
    def __getitem__(self, index):
        'Generate one batch of data'
        self.key, key1, key2 = random.split(self.key,3)
        inputs = self.__data_generation(key1,key2)
        return inputs

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key1, key2):

        pass
    
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
#         self.single_local_points = self.point_allocation(single_domains,total_points)
#         self.double_local_points = self.point_allocation(double_domains,total_points)
#         self.total_points = jnp.sum(self.single_local_points) + jnp.sum(self.double_local_points)

#         # get random points for single domains
#         single_domain_key_array = random.split(key,len(self.single_local_points))
#         single_domain_point_dict = []
#         for idx in jnp.arange(len(single_domains)):
#             temp = self.get_points(self.single_domains[idx],self.single_local_points[idx],single_domain_key_array[idx])
#             single_domain_point_dict.append(temp)
#         self.single_domain_point_dict = single_domain_point_dict   

#         # get random points for double domains
#         double_domain_key_array = random.split(key,len(self.double_local_points))
#         double_domain_point_dict = []
#         for idx in jnp.arange(len(double_domains)):
#             temp = self.get_points(self.double_domains[idx],self.double_local_points[idx],double_domain_key_array[idx])
#             temp2 = self.get_weights()
#             double_domain_point_dict.append(temp)
#         self.double_domain_point_dict = double_domain_point_dict

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