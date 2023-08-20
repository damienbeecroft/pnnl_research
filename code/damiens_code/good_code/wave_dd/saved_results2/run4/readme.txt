I ran the wave code again with all the same parameters. I just changed the linear correlation in linear_DNN to see if that helped.

ymin_A = float(sys.argv[1])
ymin_B = float(sys.argv[2])
init_lr = float(sys.argv[3]) # try 1e-2, 1e-3, 1e-4 
decay = float(sys.argv[4]) # with decay rates 0.95 and 0.99
N_nl = int(sys.argv[5]) # try 60 and 80

# ymin_A = 0.
# ymin_B = 1.
# init_lr = 1e-3
# decay = 0.99
# N_nl = 80

N_low = 100
layers = [2, N_low, N_low, N_low, N_low, N_low, 1]
layer_sizes_nl = [3,N_nl, N_nl, N_nl, N_nl, 1]
# layer_sizes_l = [1, 20, 1]
layer_sizes_l = [1,1]

a = 0.5
c = 2
batch_size = 300
batch_size_s = 300
epochs = 100000
epochsA2 = 100000

lr = optimizers.exponential_decay(init_lr, decay_steps=2000, decay_rate=decay) 
lrA = optimizers.exponential_decay(1e-4, decay_steps=2000, decay_rate=0.95)
ics_weight = 20.0
res_weight = 1.0
ut_weight = 1.0

#==== parameters that I am adding =====
delta = 1.9
#======================================

steps_to_train = np.arange(3)
reload = [False, False, False]

reloadA = False

####################################################################################################
Also, in utils_fs_v2.py in linear_DNN I set
####################################################################################################

W = np.identity(d_in)
b = np.zeros(d_out)
