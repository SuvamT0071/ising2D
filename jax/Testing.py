import runMH
import jax.numpy as jnp
import numpy as np
import corner
import matplotlib.pyplot as plt

# Testing
# Testing with a Gaussian
def log_func(x,sigma=10):
    return (-jnp.sum(x**2,axis=1))/(2*sigma**2)
num_dim = 5
num_walkers = 10
num_burn = 10000
num_samples = 100000

mh = runMH.MetropolisHastings(p0=jnp.array([[0.0,0.0,5.0,10.0,0.0]]*num_walkers),n_walkers=num_walkers,n_dim=num_dim,
                        num_samples=num_samples,num_burn=num_burn,param_ranges=jnp.array([[-10.0,10.0]]*num_dim))


samples = mh.run_mcmc(log_func)

samples = np.array(samples[num_burn:])
samples = samples.reshape(-1,num_dim)


corner.corner(np.array(samples))
plt.savefig("test.jpeg",dpi=200)
plt.close()