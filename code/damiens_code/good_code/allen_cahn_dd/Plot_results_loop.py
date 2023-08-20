#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 16:24:15 2019

@author: howa549
"""

#  
 # CS1 = plt.contour(X, Y, phi1.transpose(), levels=[0.5], colors=('#59a14f'), linestyles=('--'), linewidths=(2))
 # CS2 = plt.contour(X, Y, phi2.transpose(), levels=[0.5], colors=('#4e79a7'), linestyles=('-.'), linewidths=(2))
 # CS3 = plt.contour(X, Y, phi3.transpose(), levels=[0.5], colors=('#e15759'), linestyles=(':'), linewidths=(2))
 # CS4 = plt.contour(X, Y, phi4.transpose(), levels=[0.5], colors=('k'), linestyles=('-'), linewidths=(1.5))



import jax.numpy as np
import matplotlib
import math
import numpy as onp
import matplotlib.pyplot as plt
import scipy.io
from jax import random
from jax.experimental.ode import odeint
import jax
import matplotlib.colors as colors
if __name__ == "__main__":
    # Modified

    n_runs = 4
    
    errors = onp.zeros(n_runs + 1)

    # Tmaxes = [1.0]
    # learning_rates = [0.01,0.001,0.0001]
    # decay_rates = [0.95,0.99]
    # widths = [30,40]
    Tmaxes = [1.0]
    learning_rates = [0.01]
    decay_rates = [0.95]
    widths = [30]
    for Tmax in Tmaxes:
        for learning_rate in learning_rates:
            for decay_rate in decay_rates:
                for width in widths:
                    path = 'C:/Users/beec613/Desktop/pnnl_research/code/damiens_code/good_code/allen_cahn_dd/out_results/cahn_0_' + str(Tmax) + '_' + str(learning_rate) + '_' + str(decay_rate) + '_' + str(width) + '/'
                    # A
                    fig1, ax = plt.subplots()

                    post = 'MF_loop_res10/'
                    net_data_dirHF  = path + 'results_A/' + post
                    xmax = 1
                    xmin = 0

                    
                    data_dir = net_data_dirHF + "beta_"
                    d_vx = scipy.io.loadmat(data_dir + "test.mat")
                    t,  x, U_pred, U_star= (d_vx["t"].astype(np.float32), 
                                d_vx["x"].astype(np.float32),
                                d_vx["U_pred"].astype(np.float32),
                                d_vx["U_star"].astype(np.float32))


                        
                        
                    plt.figure(figsize=(10, 3))
                    plt.subplot(1, 3, 1)
                    plt.pcolor(t, x, U_star, cmap='jet', shading='auto', vmax=1.5, vmin=-1.5)
                    plt.xlim([xmin, xmax])

                    cbar = plt.colorbar()
                    cbar.ax.tick_params(labelsize=14) 

                    plt.xlabel('$t$', fontsize=14)
                    plt.ylabel('$x$', fontsize=14)
                    plt.title('Exact u(t, x)', fontsize=14)
                    plt.tick_params(labelsize=14)

                    plt.tight_layout()
                    
                    plt.subplot(1, 3, 2)
                    plt.pcolor(t, x, U_pred, cmap='jet', shading='auto', vmax=1.5, vmin=-1.5)
                    plt.xlim([xmin, xmax])
                    cbar = plt.colorbar()

                    plt.xlabel('$t$', fontsize=14)
                    plt.ylabel('$x$', fontsize=14)
                    plt.title('Predicted u(t, x)', fontsize=14)
                    plt.tick_params(labelsize=14)
                    cbar.ax.tick_params(labelsize=14) 

                    plt.tight_layout()
                    
                    plt.subplot(1, 3, 3)
                    Z = (np.abs(U_star - U_pred))
                    print(np.max(Z))
                    #plt.pcolor(t, x, Z, norm=colors.LogNorm(vmin=pow(10, -4), vmax=pow(10, 1)), cmap='jet', shading='auto')   
                    plt.pcolor(t, x, Z, vmax = 0.5, cmap='jet', shading='auto')    

                    plt.xlim([xmin, xmax])

                    cbar = plt.colorbar()
                    plt.xlabel('$t$', fontsize=14)
                    plt.ylabel('$x$', fontsize=14)
                    plt.title('Absolute error', fontsize=14)
                    plt.tick_params(labelsize=14)
                    cbar.ax.tick_params(labelsize=14) 

                    plt.tight_layout()

                    plt.savefig(net_data_dirHF + '/A.png', format='png')
                    
                    error_u = np.linalg.norm(U_star - U_pred, 2) / np.linalg.norm(U_star, 2)
                    errors[0] = error_u

                    print('Relative L2 error_u: %e' % (error_u))
                    
                    plt.figure(fig1.number)   
                    d_vx = scipy.io.loadmat(net_data_dirHF +'/losses.mat')
                    train, res, ics, data = ( d_vx["training_loss"].astype(np.float32),
                        d_vx["res_loss"].astype(np.float32),
                        d_vx["ics_loss"].astype(np.float32),
                        d_vx["ut_loss"].astype(np.float32))
                    
                    step = np.arange(0, 1000*len(train[0]), 1000)
                    plt.semilogy(step, train[0], label = 'Total Loss')
                    plt.semilogy(step, res[0], label = 'Residual Loss')
                    plt.semilogy(step, ics[0], label = 'IC Loss')
                    plt.semilogy(step, data[0], label = 'Data Loss')
                    plt.legend()
                    plt.savefig(net_data_dirHF + '/A_loss.png', format='png')

                #     plt.show()
                #     plt.semilogy(step, train[0], label = 'Step 0')
                    
                    
                    for i in np.arange(n_runs):
                        net_data_dirHF  = path + 'results_' + str(i) +"/" +  post
                        data_dir = net_data_dirHF + "beta_"
                        d_vx = scipy.io.loadmat(data_dir + "test.mat")
                        t,  x, U_pred= (d_vx["t"].astype(np.float32), 
                                d_vx["x"].astype(np.float32),
                                d_vx["U_pred"].astype(np.float32))


                            
                            
                        plt.figure(figsize=(10, 3))
                        plt.subplot(1, 3, 1)
                        plt.pcolor(t, x, U_star, cmap='jet', shading='auto', vmax=1.5, vmin=-1.5)
                        plt.xlim([xmin, xmax])

                        cbar = plt.colorbar()
                        cbar.ax.tick_params(labelsize=14) 

                        plt.xlabel('$t$', fontsize=14)
                        plt.ylabel('$x$', fontsize=14)
                        plt.title('Exact u(t, x)', fontsize=14)
                        plt.tick_params(labelsize=14)

                        plt.tight_layout()
                        
                        plt.subplot(1, 3, 2)
                        plt.pcolor(t, x, U_pred, cmap='jet', shading='auto', vmax=1.5, vmin=-1.5)
                        plt.xlim([xmin, xmax])
                        cbar = plt.colorbar()

                        plt.xlabel('$t$', fontsize=14)
                        plt.ylabel('$x$', fontsize=14)
                        plt.title('Predicted u(t, x)', fontsize=14)
                        plt.tick_params(labelsize=14)
                        cbar.ax.tick_params(labelsize=14) 

                        plt.tight_layout()
                        
                        plt.subplot(1, 3, 3)
                        Z = (np.abs(U_star - U_pred))
                        # plt.pcolor(t, x, Z, norm=colors.LogNorm(vmin=pow(10, -4), vmax=pow(10, 1)), cmap='jet', shading='auto')    
                        plt.pcolor(t, x, Z,  cmap='jet', shading='auto')    
                        plt.xlim([xmin, xmax])

                        cbar = plt.colorbar()
                        plt.xlabel('$t$', fontsize=14)
                        plt.ylabel('$x$', fontsize=14)
                        plt.title('Absolute error', fontsize=14)
                        plt.tick_params(labelsize=14)
                        cbar.ax.tick_params(labelsize=14) 

                        plt.tight_layout()

                        plt.savefig(net_data_dirHF + '/' + str(i+1) + '.png', format='png')
                        
                        error_u = np.linalg.norm(U_star - U_pred, 2) / np.linalg.norm(U_star, 2)

                        print('Relative L2 error_u: %e' % (error_u))
                        
                        errors[i+1] = error_u
                        
                        #    if i % 2 == 1:
                        plt.figure(42)   
                        d_vx = scipy.io.loadmat(net_data_dirHF +'/losses.mat')
                        train, res, ics, data = ( d_vx["training_loss"].astype(np.float32),
                            d_vx["res_loss"].astype(np.float32),
                            d_vx["ics_loss"].astype(np.float32),
                            d_vx["ut_loss"].astype(np.float32))
                        
                        step = np.arange(0, 1000*len(train[0]), 1000)
                        # plt.semilogy(step, train[0], label = 'Step ' + str(i+1))
                        plt.semilogy(step, train[0], label = 'Total Loss')
                        plt.semilogy(step, res[0], label = 'Residual Loss')
                        plt.semilogy(step, ics[0], label = 'IC Loss')
                        plt.semilogy(step, data[0], label = 'u_t Loss')
                        plt.legend()
                        plt.savefig(net_data_dirHF + '/' + str(i+1) + '_loss.png', format='png')
                        plt.close()
                        

                        
                        
                    
                    plt.figure(figsize=(5, 4))
                    plt.semilogy(np.arange(n_runs + 1), errors, marker='o')
                    plt.xlabel('Iteration', fontsize=14)
                    plt.ylabel('Relative L2 error', fontsize=14)
                    plt.tick_params(labelsize=14)
                    plt.tight_layout()

                    plt.savefig(net_data_dirHF + '/Errors.png', format='png')

                        
                    plt.figure(fig1.number)   
                    plt.legend(fontsize =12)

                
                
                

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Mon Aug 26 16:24:15 2019

# @author: howa549
# """

# #  
#  # CS1 = plt.contour(X, Y, phi1.transpose(), levels=[0.5], colors=('#59a14f'), linestyles=('--'), linewidths=(2))
#  # CS2 = plt.contour(X, Y, phi2.transpose(), levels=[0.5], colors=('#4e79a7'), linestyles=('-.'), linewidths=(2))
#  # CS3 = plt.contour(X, Y, phi3.transpose(), levels=[0.5], colors=('#e15759'), linestyles=(':'), linewidths=(2))
#  # CS4 = plt.contour(X, Y, phi4.transpose(), levels=[0.5], colors=('k'), linestyles=('-'), linewidths=(1.5))



# import jax.numpy as np
# import matplotlib
# import math
# import numpy as onp
# import matplotlib.pyplot as plt
# import scipy.io
# from jax import random
# from jax.experimental.ode import odeint
# import jax
# import matplotlib.colors as colors
# if __name__ == "__main__":
#     # Modified

#     n_runs = 3
    
#     errors = onp.zeros(n_runs + 1)

#     Tmaxes = [0.8,1.0,1.2]
#     for Tmax in Tmaxes:
#         path = 'C:/Users/beec613/Desktop/pnnl_research/code/damiens_code/good_code/wave_dd/out_results/wave_0_' + str(Tmax) + '/'
#         # A
#         fig1, ax = plt.subplots()

#         post = 'MF_loop/'
#         net_data_dirHF  = path + 'results_A/' + post
#         xmax = 1
#         xmin = 0

        
#         data_dir = net_data_dirHF + "beta_"
#         d_vx = scipy.io.loadmat(data_dir + "test.mat")
#         t,  x, U_pred, U_star= (d_vx["t"].astype(np.float32), 
#                     d_vx["x"].astype(np.float32),
#                     d_vx["U_pred"].astype(np.float32),
#                     d_vx["U_star"].astype(np.float32))


            
            
#         plt.figure(figsize=(10, 3))
#         plt.subplot(1, 3, 1)
#         plt.pcolor(t, x, U_star, cmap='jet', shading='auto', vmax=1.5, vmin=-1.5)
#         plt.xlim([xmin, xmax])

#         cbar = plt.colorbar()
#         cbar.ax.tick_params(labelsize=14) 

#         plt.xlabel('$t$', fontsize=14)
#         plt.ylabel('$x$', fontsize=14)
#         plt.title('Exact u(t, x)', fontsize=14)
#         plt.tick_params(labelsize=14)

#         plt.tight_layout()
        
#         plt.subplot(1, 3, 2)
#         plt.pcolor(t, x, U_pred, cmap='jet', shading='auto', vmax=1.5, vmin=-1.5)
#         plt.xlim([xmin, xmax])
#         cbar = plt.colorbar()

#         plt.xlabel('$t$', fontsize=14)
#         plt.ylabel('$x$', fontsize=14)
#         plt.title('Predicted u(t, x)', fontsize=14)
#         plt.tick_params(labelsize=14)
#         cbar.ax.tick_params(labelsize=14) 

#         plt.tight_layout()
        
#         plt.subplot(1, 3, 3)
#         Z = (np.abs(U_star - U_pred))
#         print(np.max(Z))
#         #plt.pcolor(t, x, Z, norm=colors.LogNorm(vmin=pow(10, -4), vmax=pow(10, 1)), cmap='jet', shading='auto')   
#         plt.pcolor(t, x, Z, vmax = 0.5, cmap='jet', shading='auto')    

#         plt.xlim([xmin, xmax])

#         cbar = plt.colorbar()
#         plt.xlabel('$t$', fontsize=14)
#         plt.ylabel('$x$', fontsize=14)
#         plt.title('Absolute error', fontsize=14)
#         plt.tick_params(labelsize=14)
#         cbar.ax.tick_params(labelsize=14) 

#         plt.tight_layout()

#         plt.savefig(net_data_dirHF + '/A.png', format='png')
        
#         error_u = np.linalg.norm(U_star - U_pred, 2) / np.linalg.norm(U_star, 2)
#         errors[0] = error_u

#         print('Relative L2 error_u: %e' % (error_u))
        
#         plt.figure(fig1.number)   
#         d_vx = scipy.io.loadmat(net_data_dirHF +'/losses.mat')
#         train, res, ics, data = ( d_vx["training_loss"].astype(np.float32),
#             d_vx["res_loss"].astype(np.float32),
#             d_vx["ics_loss"].astype(np.float32),
#             d_vx["ut_loss"].astype(np.float32))
        
#         step = np.arange(0, 1000*len(train[0]), 1000)
#         plt.semilogy(step, train[0], label = 'Total Loss')
#         plt.semilogy(step, res[0], label = 'Residual Loss')
#         plt.semilogy(step, ics[0], label = 'IC Loss')
#         plt.semilogy(step, data[0], label = 'Data Loss')
#         plt.legend()
#         plt.savefig(net_data_dirHF + '/A_loss.png', format='png')

#     #     plt.show()
#     #     plt.semilogy(step, train[0], label = 'Step 0')
        
        
#         for i in np.arange(n_runs):
#             net_data_dirHF  = path + 'results_' + str(i) +"/" +  post
#             data_dir = net_data_dirHF + "beta_"
#             d_vx = scipy.io.loadmat(data_dir + "test.mat")
#             t,  x, U_pred= (d_vx["t"].astype(np.float32), 
#                     d_vx["x"].astype(np.float32),
#                     d_vx["U_pred"].astype(np.float32))


                
                
#             plt.figure(figsize=(10, 3))
#             plt.subplot(1, 3, 1)
#             plt.pcolor(t, x, U_star, cmap='jet', shading='auto', vmax=1.5, vmin=-1.5)
#             plt.xlim([xmin, xmax])

#             cbar = plt.colorbar()
#             cbar.ax.tick_params(labelsize=14) 

#             plt.xlabel('$t$', fontsize=14)
#             plt.ylabel('$x$', fontsize=14)
#             plt.title('Exact u(t, x)', fontsize=14)
#             plt.tick_params(labelsize=14)

#             plt.tight_layout()
            
#             plt.subplot(1, 3, 2)
#             plt.pcolor(t, x, U_pred, cmap='jet', shading='auto', vmax=1.5, vmin=-1.5)
#             plt.xlim([xmin, xmax])
#             cbar = plt.colorbar()

#             plt.xlabel('$t$', fontsize=14)
#             plt.ylabel('$x$', fontsize=14)
#             plt.title('Predicted u(t, x)', fontsize=14)
#             plt.tick_params(labelsize=14)
#             cbar.ax.tick_params(labelsize=14) 

#             plt.tight_layout()
            
#             plt.subplot(1, 3, 3)
#             Z = (np.abs(U_star - U_pred))
#             # plt.pcolor(t, x, Z, norm=colors.LogNorm(vmin=pow(10, -4), vmax=pow(10, 1)), cmap='jet', shading='auto')    
#             plt.pcolor(t, x, Z,  cmap='jet', shading='auto')    
#             plt.xlim([xmin, xmax])

#             cbar = plt.colorbar()
#             plt.xlabel('$t$', fontsize=14)
#             plt.ylabel('$x$', fontsize=14)
#             plt.title('Absolute error', fontsize=14)
#             plt.tick_params(labelsize=14)
#             cbar.ax.tick_params(labelsize=14) 

#             plt.tight_layout()

#             plt.savefig(net_data_dirHF + '/' + str(i+1) + '.png', format='png')
            
#             error_u = np.linalg.norm(U_star - U_pred, 2) / np.linalg.norm(U_star, 2)

#             print('Relative L2 error_u: %e' % (error_u))
            
#             errors[i+1] = error_u
            
#             #    if i % 2 == 1:
#             plt.figure(42)   
#             d_vx = scipy.io.loadmat(net_data_dirHF +'/losses.mat')
#             train, res, ics, data = ( d_vx["training_loss"].astype(np.float32),
#                 d_vx["res_loss"].astype(np.float32),
#                 d_vx["ics_loss"].astype(np.float32),
#                 d_vx["ut_loss"].astype(np.float32))
            
#             step = np.arange(0, 1000*len(train[0]), 1000)
#             # plt.semilogy(step, train[0], label = 'Step ' + str(i+1))
#             plt.semilogy(step, train[0], label = 'Total Loss')
#             plt.semilogy(step, res[0], label = 'Residual Loss')
#             plt.semilogy(step, ics[0], label = 'IC Loss')
#             plt.semilogy(step, data[0], label = 'u_t Loss')
#             plt.legend()
#             plt.savefig(net_data_dirHF + '/' + str(i+1) + '_loss.png', format='png')
#             plt.close()
            

            
            
        
#         plt.figure(figsize=(5, 4))
#         plt.semilogy(np.arange(n_runs + 1), errors, marker='o')
#         plt.xlabel('Iteration', fontsize=14)
#         plt.ylabel('Relative L2 error', fontsize=14)
#         plt.tick_params(labelsize=14)
#         plt.tight_layout()

#         plt.savefig(net_data_dirHF + '/Errors.png', format='png')

            
#         plt.figure(fig1.number)   
#         plt.legend(fontsize =12)

    
    
    


# # #!/usr/bin/env python3
# # # -*- coding: utf-8 -*-
# # """
# # Created on Mon Aug 26 16:24:15 2019

# # @author: howa549
# # """

# # #  
# #  # CS1 = plt.contour(X, Y, phi1.transpose(), levels=[0.5], colors=('#59a14f'), linestyles=('--'), linewidths=(2))
# #  # CS2 = plt.contour(X, Y, phi2.transpose(), levels=[0.5], colors=('#4e79a7'), linestyles=('-.'), linewidths=(2))
# #  # CS3 = plt.contour(X, Y, phi3.transpose(), levels=[0.5], colors=('#e15759'), linestyles=(':'), linewidths=(2))
# #  # CS4 = plt.contour(X, Y, phi4.transpose(), levels=[0.5], colors=('k'), linestyles=('-'), linewidths=(1.5))



# # import jax.numpy as np
# # import matplotlib
# # import math
# # import numpy as onp
# # import matplotlib.pyplot as plt
# # import scipy.io
# # from jax import random
# # from jax.experimental.ode import odeint
# # import jax
# # import matplotlib.colors as colors
# # if __name__ == "__main__":
# #     # Modified

# #     n_runs = 2
    
# #     errors = onp.zeros(n_runs + 1)
# # #     C:\Users\beec613\Desktop\pnnl_research\code\damiens_code\good_code\wave_dd\out_results\wave_0_1\results_A\MF_loop\losses.mat
# #     path = 'C:/Users/beec613/Desktop/pnnl_research/code/damiens_code/good_code/wave_dd/out_results/wave_0_1/'
# #     # A
# #     fig1, ax = plt.subplots()

# #     post = 'MF_loop/'
# #     net_data_dirHF  = path + 'results_A/' + post
# #     xmax = 1
# #     xmin = 0

    
# #     data_dir = net_data_dirHF + "beta_"
# #     d_vx = scipy.io.loadmat(data_dir + "test.mat")
# #     t,  x, U_pred, U_star= (d_vx["t"].astype(np.float32), 
# #                 d_vx["x"].astype(np.float32),
# #                 d_vx["U_pred"].astype(np.float32),
# #                 d_vx["U_star"].astype(np.float32))


        
        
# #     plt.figure(figsize=(10, 3))
# #     plt.subplot(1, 3, 1)
# #     plt.pcolor(t, x, U_star, cmap='jet', shading='auto', vmax=1.5, vmin=-1.5)
# #     plt.xlim([xmin, xmax])

# #     cbar = plt.colorbar()
# #     cbar.ax.tick_params(labelsize=14) 

# #     plt.xlabel('$t$', fontsize=14)
# #     plt.ylabel('$x$', fontsize=14)
# #     plt.title('Exact u(t, x)', fontsize=14)
# #     plt.tick_params(labelsize=14)

# #     plt.tight_layout()
    
# #     plt.subplot(1, 3, 2)
# #     plt.pcolor(t, x, U_pred, cmap='jet', shading='auto', vmax=1.5, vmin=-1.5)
# #     plt.xlim([xmin, xmax])
# #     cbar = plt.colorbar()

# #     plt.xlabel('$t$', fontsize=14)
# #     plt.ylabel('$x$', fontsize=14)
# #     plt.title('Predicted u(t, x)', fontsize=14)
# #     plt.tick_params(labelsize=14)
# #     cbar.ax.tick_params(labelsize=14) 

# #     plt.tight_layout()
    
# #     plt.subplot(1, 3, 3)
# #     Z = (np.abs(U_star - U_pred))
# #     print(np.max(Z))
# #     #plt.pcolor(t, x, Z, norm=colors.LogNorm(vmin=pow(10, -4), vmax=pow(10, 1)), cmap='jet', shading='auto')   
# #     plt.pcolor(t, x, Z, vmax = 0.5, cmap='jet', shading='auto')    

# #     plt.xlim([xmin, xmax])

# #     cbar = plt.colorbar()
# #     plt.xlabel('$t$', fontsize=14)
# #     plt.ylabel('$x$', fontsize=14)
# #     plt.title('Absolute error', fontsize=14)
# #     plt.tick_params(labelsize=14)
# #     cbar.ax.tick_params(labelsize=14) 

# #     plt.tight_layout()

# #     plt.savefig(net_data_dirHF + '/A.png', format='png')
    
# #     error_u = np.linalg.norm(U_star - U_pred, 2) / np.linalg.norm(U_star, 2)
# #     errors[0] = error_u

# #     print('Relative L2 error_u: %e' % (error_u))
    
# #     plt.figure(fig1.number)   
# #     d_vx = scipy.io.loadmat(net_data_dirHF +'/losses.mat')
# #     train, res, ics, data = ( d_vx["training_loss"].astype(np.float32),
# #          d_vx["res_loss"].astype(np.float32),
# #          d_vx["ics_loss"].astype(np.float32),
# #          d_vx["ut_loss"].astype(np.float32))
    
# #     step = np.arange(0, 1000*len(train[0]), 1000)
# #     plt.semilogy(step, train[0], label = 'Total Loss')
# #     plt.semilogy(step, res[0], label = 'Residual Loss')
# #     plt.semilogy(step, ics[0], label = 'IC Loss')
# #     plt.semilogy(step, data[0], label = 'Data Loss')
# #     plt.legend()
# #     plt.savefig(net_data_dirHF + '/A_loss.png', format='png')

# # #     plt.show()
# # #     plt.semilogy(step, train[0], label = 'Step 0')
    
    
# #     for i in np.arange(n_runs):
# #      net_data_dirHF  = path + 'results_' + str(i) +"/" +  post
# #      data_dir = net_data_dirHF + "beta_"
# #      d_vx = scipy.io.loadmat(data_dir + "test.mat")
# #      t,  x, U_pred= (d_vx["t"].astype(np.float32), 
# #                d_vx["x"].astype(np.float32),
# #                d_vx["U_pred"].astype(np.float32))


          
          
# #      plt.figure(figsize=(10, 3))
# #      plt.subplot(1, 3, 1)
# #      plt.pcolor(t, x, U_star, cmap='jet', shading='auto', vmax=1.5, vmin=-1.5)
# #      plt.xlim([xmin, xmax])

# #      cbar = plt.colorbar()
# #      cbar.ax.tick_params(labelsize=14) 

# #      plt.xlabel('$t$', fontsize=14)
# #      plt.ylabel('$x$', fontsize=14)
# #      plt.title('Exact u(t, x)', fontsize=14)
# #      plt.tick_params(labelsize=14)

# #      plt.tight_layout()
     
# #      plt.subplot(1, 3, 2)
# #      plt.pcolor(t, x, U_pred, cmap='jet', shading='auto', vmax=1.5, vmin=-1.5)
# #      plt.xlim([xmin, xmax])
# #      cbar = plt.colorbar()

# #      plt.xlabel('$t$', fontsize=14)
# #      plt.ylabel('$x$', fontsize=14)
# #      plt.title('Predicted u(t, x)', fontsize=14)
# #      plt.tick_params(labelsize=14)
# #      cbar.ax.tick_params(labelsize=14) 

# #      plt.tight_layout()
     
# #      plt.subplot(1, 3, 3)
# #      Z = (np.abs(U_star - U_pred))
# #      # plt.pcolor(t, x, Z, norm=colors.LogNorm(vmin=pow(10, -4), vmax=pow(10, 1)), cmap='jet', shading='auto')    
# #      plt.pcolor(t, x, Z,  cmap='jet', shading='auto')    
# #      plt.xlim([xmin, xmax])

# #      cbar = plt.colorbar()
# #      plt.xlabel('$t$', fontsize=14)
# #      plt.ylabel('$x$', fontsize=14)
# #      plt.title('Absolute error', fontsize=14)
# #      plt.tick_params(labelsize=14)
# #      cbar.ax.tick_params(labelsize=14) 

# #      plt.tight_layout()

# #      plt.savefig(net_data_dirHF + '/' + str(i+1) + '.png', format='png')
     
# #      error_u = np.linalg.norm(U_star - U_pred, 2) / np.linalg.norm(U_star, 2)

# #      print('Relative L2 error_u: %e' % (error_u))
     
# #      errors[i+1] = error_u
     
# #      #    if i % 2 == 1:
# #      plt.figure(42)   
# #      d_vx = scipy.io.loadmat(net_data_dirHF +'/losses.mat')
# #      train, res, ics, data = ( d_vx["training_loss"].astype(np.float32),
# #           d_vx["res_loss"].astype(np.float32),
# #           d_vx["ics_loss"].astype(np.float32),
# #           d_vx["ut_loss"].astype(np.float32))
     
# #      step = np.arange(0, 1000*len(train[0]), 1000)
# #      # plt.semilogy(step, train[0], label = 'Step ' + str(i+1))
# #      plt.semilogy(step, train[0], label = 'Total Loss')
# #      plt.semilogy(step, res[0], label = 'Residual Loss')
# #      plt.semilogy(step, ics[0], label = 'IC Loss')
# #      plt.semilogy(step, data[0], label = 'u_t Loss')
# #      plt.legend()
# #      plt.savefig(net_data_dirHF + '/' + str(i+1) + '_loss.png', format='png')
# #      plt.close()
        

        
        
    
# #     plt.figure(figsize=(5, 4))
# #     plt.semilogy(np.arange(n_runs + 1), errors, marker='o')
# #     plt.xlabel('Iteration', fontsize=14)
# #     plt.ylabel('Relative L2 error', fontsize=14)
# #     plt.tick_params(labelsize=14)
# #     plt.tight_layout()

# #     plt.savefig(net_data_dirHF + '/Errors.png', format='png')

        
# #     plt.figure(fig1.number)   
# #     plt.legend(fontsize =12)

    
    
    