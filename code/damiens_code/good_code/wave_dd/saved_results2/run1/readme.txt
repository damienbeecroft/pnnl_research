I ran the code with the following parameters. It did not converge well, so I am going to change up the parameters and run it again.

N_low = 100
layers = [2, N_low, N_low, N_low, N_low, N_low, 1]
N_low=120
layer_sizes_nl = [3,N_low, N_low, N_low, N_low, N_low, 1]
# layer_sizes_l = [1,20, 1]
layer_sizes_l = [1,1]

a = 0.5
c = 2
batch_size = 300
batch_size_s = 300
epochs = 100000
epochsA2 = 100000
lr = optimizers.exponential_decay(1e-3, decay_steps=2000, decay_rate=0.99)
lrA = optimizers.exponential_decay(1e-4, decay_steps=2000, decay_rate=0.95)
ics_weight = 20.0
res_weight = 1.0
ut_weight = 1

# ymin_A = 0.
# ymin_B = 1.
ymin_A = float(sys.argv[1])
ymin_B = float(sys.argv[2])
# print("ymin_A: %.3f" % ymin_A)
#==== parameters that I am adding =====
delta = 1.9
#======================================

####################################################################################################
Also, in utils_fs_v2.py in linear_DNN I set
####################################################################################################

W = np.identity(d_in)
b = np.zeros(d_out)