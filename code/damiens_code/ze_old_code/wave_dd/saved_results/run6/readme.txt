I tried changing the initial condition weights to be more like the recent code that Amanda sent. This made the
solution look like a simple wave as opposed to the true solution.

    N_low = 500
    layers = [2, N_low, N_low, N_low, N_low, N_low, 1]
    N_low= 200
    layer_sizes_nl = [3, N_low, N_low, N_low, 1]
    layer_sizes_l = [1, 1]
    
    a = 0.5
    c = 2
    # batch_size = 300
    batch_size = 1000
    batch_size_s = 300
    epochs = 100000
    # epochsA2 = 10
    epochsA2 = 200000
    lr = optimizers.exponential_decay(5e-4, decay_steps=2000, decay_rate=0.99)
    lrA = optimizers.exponential_decay(1e-4, decay_steps=2000, decay_rate=0.99)
    ics_weight = 20.
    res_weight = 1.
    ut_weight = 1.


    ymin_A = 0.0
    ymin_B = 1.0

    #==== parameters that I am adding =====
    delta = 1.9
    #======================================

    steps_to_train = np.arange(1)
    reload = [False]
    
    reloadA = True