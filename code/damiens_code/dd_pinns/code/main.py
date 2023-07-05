
# Import operating system
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Their imports from other packages
from jax.example_libraries import optimizers
import scipy.io
import jax
import jax.numpy as np
from jax.flatten_util import ravel_pytree

# My imports from other packages
# import dill
# import pickle

# Imports from local files
from utils_fs_v2 import  DataGenerator, DataGenerator_res
from SFDomainNet_Class import DomainNet

# def save_neural_domain_tree(NDTree,path):
#     save_SFDomainNet(NDTree[0],)
#     for level in NDTree:
#         accumulator
#         for network in level:
#             net

def save_MFDomainNet(model, save_results_to, save_prfx):
    # ====================================
    # Saving model
    # ====================================
    params = model.get_params(model.opt_state)
    t_train_range = np.linspace(0, 50, 2000)
    u_res = t_train_range.reshape([len(t_train_range), 1])
    flat_params, _  = ravel_pytree(model.get_params(model.opt_state))
    np.save(save_results_to + 'params_' + save_prfx + '.npy', flat_params)

    S_pred =  model.predict_full(params, u_res)

    fname= save_results_to + "beta_test.mat"
    scipy.io.savemat(fname, {'U_res':u_res,'S_pred':S_pred})
    
    scipy.io.savemat(save_results_to +"losses.mat", 
                     {'training_loss':model.loss_training_log,
                      'res_loss':model.loss_res_log,
                      'ics_loss':model.loss_ics_log,
                      'data_loss':model.loss_data_log})

def save_SFDomainNet(model, save_results_to, save_prfx):
    # ===================================
    # Saving model
    # ====================================
    params = model.get_params(model.opt_state)
    t_train_range = np.linspace(0, 50, 2000)
    u_res = t_train_range.reshape([len(t_train_range), 1])
    flat_params, _  = ravel_pytree(model.get_params(model.opt_state))
    np.save(save_results_to + 'params_' + save_prfx + '.npy', flat_params)

    S_pred =  model.predict_low(params, u_res)

    fname= save_results_to +"beta_test.mat"
    scipy.io.savemat(fname, {'U_res':u_res,'S_pred':S_pred})
    
    scipy.io.savemat(save_results_to +"losses.mat", 
                     {'training_loss':model.loss_training_log,
                      'res_loss':model.loss_res_log,
                      'ics_loss':model.loss_ics_log,
                      'data_loss':model.loss_data_log})




if __name__=="__main__":
    # weights for neural net loss function
    ics_weight = 1.0 # initial conditions penalty
    res_weight = 1.0 # residual penalty
    data_weight  = 0.0 
    pen_weight  = 0.000001 # regularization penalty

    # batch sizes
    # NOTE: Why are there two batch sizes?
    batch_size = 100
    batch_size_res = int(batch_size/2) 

    # Variables that define the probability distribution of the collocation points. 
    # A detailed definition is given in "A comprehensive study of non-adaptive and residual-based adaptive
    # sampling for physics-informed neural networks", equation 2.
    k = 2
    c = 0
    
    epochs = 1000 # number of epochs for the first, single-fidelity PINN
    epochsA2 = 100000 # number of epochs for subsequent multi-fidelity PINNs
    lr = optimizers.exponential_decay(1e-3, decay_steps=2000, decay_rate=0.99)
    N_low = 200 # low fidelity hidden width
    N_nl = 80 # nonlinear network hidden width
    layers_A = [1, N_low, N_low, N_low, 2]
    layers_sizes_nl = [3, N_nl, N_nl, N_nl, 2]
    layers_sizes_l = [2,  4, 2]

    min_A = 0 # lower solution domain boundary
    min_B = 10 # upper solution domain boundary
    Tmax = min_B 
    delta = 1.9 # variable that determines the amount of overlap of domains

    data_range = np.arange(0,int(2*min_B)) # NOTE: What is the purpose of this?

    
    d_vx = scipy.io.loadmat("C:/Users/beec613/Desktop/pnnl_research/code/damiens_code/dd_pinns/code/input/data.mat") # NOTE: Need to make this OS agnostic at some point
    t_data_full, s_data_full = (d_vx["u"].astype(np.float32), d_vx["s"].astype(np.float32))

    # ====================================
    # Saving Settings
    # ====================================
    output_dir = "C:/Users/beec613/Desktop/pnnl_research/code/damiens_code/dd_pinns/code/output/" # NOTE: Need to make this OS agnostic at some point 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # ====================================
    # Train A
    # ====================================
    
    u_bc = np.asarray([0]).reshape([1, -1]) # time where the boundary condition is imposed
    s_bc = np.asarray([1, 1]).reshape([1, -1]) # pendulum starts with s1=1, s2=1
    u_bc = jax.device_put(u_bc)
    s_bc = jax.device_put(s_bc)

    t_data = jax.device_put(t_data_full[:, data_range].reshape([-1, 1]))
    s_data = jax.device_put(s_data_full[data_range, :].reshape([-1, 2]))

    # create data set
    coords = [min_A, min_B]
    ic_dataset = DataGenerator(u_bc, s_bc, 1)
    res_dataset = DataGenerator_res(coords, batch_size)
    data_dataset = DataGenerator(t_data, s_data, len(t_data))

    lam = []
    F = []
    results_dir = output_dir
    
    model_A = DomainNet(layers_A, ics_weight, res_weight, data_weight, [], lr,[[min_A],[min_B]])
    model_A.train(ic_dataset, res_dataset, data_dataset, lam, F, [], nIter=epochs)

    NDTree = []
    NDTree.append(model_A)

    print("hello")


#     model_A = DNN_class_EWC(layers_A, ics_weight, res_weight, data_weight, [], lr)
#     if reloadA:
#         # params_A = model_A.unravel_params(np.load(results_dir+ '/params.npy')) # This was the original line
#         params_A = model_A.unravel_params(np.load(results_dir + 'params.npy'))

    
#     else:
#         model_A.train(ic_dataset, res_dataset, data_dataset, lam, F, [], nIter=epochs)
#         params_A = model_A.get_params(model_A.opt_state)
#         save_dataDNN(model_A, params_A,  results_dir +"/", 'A')   
        
#         flat_params, _  = ravel_pytree(model_A.get_params(model_A.opt_state))
#         np.save(results_dir + 'params.npy', flat_params)
#         print('/n ... A Training done ...')
        
#         scipy.io.savemat(results_dir +"losses.mat", 
#                      {'training_loss':model_A.loss_training_log,
#                       'res_loss':model_A.loss_res_log,
#                       'ics_loss':model_A.loss_ics_log,
#                       'ut_loss':model_A.loss_data_log}, format='4')
    

#     # ====================================
#     # DNN model A2
#     # ====================================
#   #  res_weight = 100.0
#     params_prev = []
    

#     k = 2
#     c = 0 
#     key = random.PRNGKey(1234)
#     batch_size_res = int(batch_size/2)    
#     batch_size_pts = batch_size - batch_size_res                            
                                     
    
#     key, subkey = random.split(key) 

#     # These lines are computing the residuals at points based on a density function and then
#     # they compute the error to backpropagate the parameters
#     res_pts = coords[0] + (coords[1]-coords[0])*random.uniform(key, shape=[20000,1])
#     res_val = model_A.predict_res(params_A, res_pts)
#     err = res_val**k/np.mean( res_val**k) + c
#     err_norm = err/np.sum(err)                        
#     res_dataset = DataGenerator_res2(coords, res_pts, err_norm, batch_size_res, batch_size)
    
    
    
#     for step in steps_to_train:
#         results_dir = "../results_" + str(step) + "/"+save_str+"/"
#         if not os.path.exists(results_dir):
#             os.makedirs(results_dir)
        
#         if step == 0:
#             res = 0
#         else:
#             res = 1
 
#         model = MF_class_EWC(layers_sizes_nl, layers_sizes_l, layers_A, ics_weight, 
#                          res_weight, data_weight, pen_weight,lr,  params_A, params_t = params_prev, restart =res)

        
#         if reload[step]:
#             # params = model.unravel_params(np.load(results_dir + '/params.npy')) # this was the original line
#             params = model.unravel_params(np.load(results_dir + 'params.npy'))
        
#         else:     
#             model.train(ic_dataset, res_dataset, data_dataset, nIter=epochsA2)

#             print('/n ... A2 Training done ...')
#             scipy.io.savemat(results_dir +"losses.mat", 
#                          {'training_loss':model.loss_training_log,
#                           'res_loss':model.loss_res_log,
#                           'ics_loss':model.loss_ics_log,
#                           'ut_loss':model.loss_data_log})
        
#             params = model.get_params(model.opt_state)
#             flat_params, _  = ravel_pytree(params)
#             np.save(results_dir + 'params.npy', flat_params)
        
#             save_data(model,  params, results_dir, 'B')   
            
#         params_prev.append(params)
        
#         key, subkey = random.split(key)
#         res_pts = coords[0] + (coords[1]-coords[0])*random.uniform(key, shape=[20000,1])
#         res_val = model.predict_res(params, res_pts)
#         err = res_val**k/np.mean( res_val**k) + c
#         err_norm = err/np.sum(err)                        
      
#         res_dataset = DataGenerator_res2(coords, res_pts, err_norm, batch_size_res, batch_size)


#####################################################################################################
    #====================================================================================
    # Load the root
    #====================================================================================




    #====================================================================================
    # The rest of this is mutli-fidelity. Ignore it for now.
    #====================================================================================

    # params_prev = []
    

    # k = 2
    # c = 0 
    # key = random.PRNGKey(1234)
    # batch_size_res = int(batch_size/2)    
    # batch_size_pts = batch_size - batch_size_res                            
                                     
    
    # key, subkey = random.split(key)

    # res_pts = coords[0] + (coords[1]-coords[0])*random.uniform(key, shape=[20000,1])
    # res_val = model_A.predict_res(params_A, res_pts)
    # err = res_val**k/np.mean( res_val**k) + c
    # err_norm = err/np.sum(err)                        
    # res_dataset = DataGenerator_res2(coords, res_pts, err_norm, batch_size_res, batch_size)
    
    
    # Ndomains = []
    # for step in steps_to_train:
    #     # results_dir = "../results_" + str(step) + "/"+save_str+"/" # This is the original line
    #     results_dir = "C:/Users/beec613/Desktop/pnnl_research/code/amandas_code/Pendulum_DD/results_" + str(step) + "/"+save_str+"/"
    #     if not os.path.exists(results_dir):
    #         os.makedirs(results_dir)
        
    #     res = 0
    #     if step > 0:
    #         res=1
            
    #     Ndomains.append(2**(step+1))
 
    #     model = MF_class_EWC(layers_sizes_nl, layers_sizes_l, layers_A, ics_weight, 
    #                      res_weight, data_weight, pen_weight,lr, Ndomains, delta, Tmax, params_A, params_t = params_prev, restart =res)

        
    #     if reload[step]:
    #         params = model.unravel_params(np.load(results_dir + '/params.npy'))

        
    #     else:     
    #         model.train(ic_dataset, res_dataset, data_dataset, nIter=epochsA2)
        



    #         print('\n ... Level ' + str(step) + ' Training done ...')
    #         scipy.io.savemat(results_dir +"losses.mat", 
    #                      {'training_loss':model.loss_training_log,
    #                       'res_loss':model.loss_res_log,
    #                       'ics_loss':model.loss_ics_log,
    #                       'ut_loss':model.loss_data_log})
        
    #         params = model.get_params(model.opt_state)
    #         flat_params, _  = ravel_pytree(params)
    #         np.save(results_dir + 'params.npy', flat_params)
        
    #         save_data(model,  params, results_dir, 'B')   
            
    #     params_prev.append(params)
        
    #     key, subkey = random.split(key)
    #     res_pts = coords[0] + (coords[1]-coords[0])*random.uniform(key, shape=[20000,1])
    #     res_val = model.predict_res(params, res_pts)
    #     err = res_val**k/np.mean( res_val**k) + c
    #     err_norm = err/np.sum(err)                        
      
    #     res_dataset = DataGenerator_res2(coords, res_pts, err_norm, batch_size_res, batch_size)