In this run I changed the linear network to initialize as the identity operator. This means that the initial approximation of the multifidelity
method is simple the low fidelity method plus whatever the nonlinear net spits out. However, the solution did not train better after the initialization.
I am trying to figure out why this is the case.
