I ran this test with the following input

ymin_A = float(sys.argv[1])
ymin_B = float(sys.argv[2])
init_lr = float(sys.argv[3]) # try 1e-2, 1e-3, 1e-4 
decay = float(sys.argv[4]) # with decay rates 0.95 and 0.99
N_nl = int(sys.argv[5]) # try 60 and 80

# ymin_A = 0.0
# ymin_B = 1.0
# init_lr = 1e-3
# decay = 0.99
# N_nl = 100

batch_size = 500 #500
batch_size_res = 0
# batch_size_res = int(batch_size/2)
Npts = 5000


N_low =200
layers = [2, N_low, N_low, N_low, N_low, N_low, 1]
layer_sizes_nl = [3, N_nl, N_nl, N_nl, 1]
# layer_sizes_l = [1, 20, 1]
layer_sizes_l = [1, 1]

batch_size_s = 100
epochs = 100000
epochsA= 100000
lr = optimizers.exponential_decay(init_lr, decay_steps=2000, decay_rate=decay)
lrA = optimizers.exponential_decay(1e-4, decay_steps=2000, decay_rate=0.99)
ics_weight = 1.0 # was 10
res_weight = 1.0
ut_weight = 1.0
energy_weight = 0. # I changed this to 0 from 1 because I don't know what c and a are


####################################################################################################
Also, in utils_fs_v2.py in linear_DNN I set
####################################################################################################

W = np.identity(d_in)
b = np.zeros(d_out)