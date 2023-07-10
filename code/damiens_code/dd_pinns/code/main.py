# Import operating system
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Their imports from other packages
import scipy.io
import jax
import jax.numpy as np
from jax.flatten_util import ravel_pytree
from jax.example_libraries import optimizers

# My imports from other packages
import matplotlib.pyplot as plt

# Imports from local files
from utils_fs_v2 import  DataGenerator, DataGenerator_res
from SFDomainNet_Class import SFDomainNet
from MFDomainNet_Class import MFDomainNet

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
    data_weight  = 0.0 # interpolation penalty
    pen_weight  = 0.000001 # regularization penalty

    # batch sizes
    batch_size = 100
    batch_size_res = int(batch_size/2) 

    # Variables that define the probability distribution of the collocation points. 
    # A detailed definition is given in "A comprehensive study of non-adaptive and residual-based adaptive
    # sampling for physics-informed neural networks", equation 2.
    k = 2
    c = 0
    
    epochs_sf = 1000 # number of epochs for the first, single-fidelity PINN
    epochs_mf = 100000 # number of epochs for subsequent multi-fidelity PINNs
    lr = optimizers.exponential_decay(1e-3, decay_steps=2000, decay_rate=0.99)
    N_sf = 200 # single fidelity hidden width
    N_nl = 80 # multifidelity nonlinear network hidden width
    sfnet_shape = [1, N_sf, N_sf, N_sf, 2] # shape of the single fidelity network
    nonlin_mfnet_shape = [3, N_nl, N_nl, N_nl, 2] # shape of the multifidelity nonlinear network
    lin_mfnet_shape = [2,  4, 2] # shape of the multifidelity linear network

    min_A = 0.0 # lower solution domain boundary
    min_B = 1.0 # upper solution domain boundary
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
    # Define Neural Domain Tree
    # ====================================
    
    t_bc = np.asarray([0]).reshape([1, -1]) # time where the boundary condition is imposed
    u_bc = np.asarray([1, 1]).reshape([1, -1]) # pendulum starts with s1=1, s2=1
    t_bc = jax.device_put(t_bc)
    u_bc = jax.device_put(u_bc)

    t_data = jax.device_put(t_data_full[:, data_range].reshape([-1, 1]))
    s_data = jax.device_put(s_data_full[data_range, :].reshape([-1, 2]))

    # create data set
    coords = [min_A, min_B]
    ic_dataset = DataGenerator(t_bc, u_bc, 1)
    res_dataset = DataGenerator_res(coords, batch_size)
    data_dataset = DataGenerator(t_data, s_data, len(t_data))

    lam = []
    F = []
    results_dir = output_dir
    
    # NOTE: I want to make it so that many of the variables that need to be passed into MFDomainNet
    #       are stored in the super. That way, they don't need to be passed in every single time.
    A = SFDomainNet(sfnet_shape, ics_weight, res_weight, data_weight, [], lr,[[min_A],[min_B]])
    B = MFDomainNet(nonlin_mfnet_shape, lin_mfnet_shape, ics_weight, res_weight, data_weight, 
                 pen_weight, lr, [], [[0.0],[0.6]], parent = A)
    C = MFDomainNet(nonlin_mfnet_shape, lin_mfnet_shape, ics_weight, res_weight, data_weight, 
                 pen_weight, lr, [], [[0.4],[1.0]], parent = A)
    D = MFDomainNet(nonlin_mfnet_shape, lin_mfnet_shape, ics_weight, res_weight, data_weight, 
                 pen_weight, lr, [], [[0.0],[0.3]], parent = B)
    E = MFDomainNet(nonlin_mfnet_shape, lin_mfnet_shape, ics_weight, res_weight, data_weight, 
                 pen_weight, lr, [], [[0.2],[0.5]], parent = B)
    F = MFDomainNet(nonlin_mfnet_shape, lin_mfnet_shape, ics_weight, res_weight, data_weight, 
                 pen_weight, lr, [], [[0.4],[0.7]], parent = C)
    G = MFDomainNet(nonlin_mfnet_shape, lin_mfnet_shape, ics_weight, res_weight, data_weight, 
                 pen_weight, lr, [], [[0.6],[1.0]], parent = C)
    
    pts = np.linspace(0.0,1.0,11)
    
    u_preds = A.evaluate_neural_domain_tree(pts)
    print("done")
    plt.plot(pts,u_preds[-1])
    plt.show()