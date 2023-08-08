Pendulum Code:

ze_old/mffbpinns contains the code that creates a batch for every relevant subdomain of the finest layer of the network. This code works, but is
incredibly slow.

ze_old/domain_tree_code is the original code that I was working on. The batch would be passed to the neural domain tree and batch points would
be sorted and distributed to the appropriate nodes. I could not get this code to work with jax.

Pendulum_DD contains a slightly modified version of Amanda's code that actually works. All neural networks are
evaluated everywhere.

Wave Code:

wave_test was the first, non-functioning code that Amanda sent to me. This code does not converge because the parameters are wrong.

wave_dd was my first attempt at the domain decomposition algorithm. I changed the parameters to have those of the correct code that Amanda sent to
me. This worked somewhat. However, I realized that I was doing the domain decomposition incorrectly.

Other:

test contains a variety of jupyter notebooks with tests so that I could understand some of the
classes and functions I was developing without having to mess with everything in the Pendulum_DD code