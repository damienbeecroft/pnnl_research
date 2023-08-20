I implemented causal penalization in this code. It seems to do as well as the non-causal method for T=[0,20]. This is not good enough

ics_weight = 1.0
res_weight = 1.0 
data_weight  = 0.0
pen_weight  = 0.000001

batch_size = 100
batch_size_res = int(batch_size/2)

steps_to_train = np.arange(3)
reload = [False, False, False]

reloadA = False

k = 2
c = 0 

epochs = 10000
epochsA2 = 100000
lr = optimizers.exponential_decay(1e-3, decay_steps=2000, decay_rate=0.99)
N_low = 100
N_nl = 80
layers_A = [1, N_low, N_low, N_low, N_low, N_low, 2]
layers_sizes_nl = [3, N_nl, N_nl, N_nl, 2]
layers_sizes_l = [2,  4, 2]
min_A = 0
min_B = 20
Tmax = min_B
delta = 1.9
epsilon = 1e-5