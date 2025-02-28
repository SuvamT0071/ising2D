import jax.numpy as jnp
from functools import partial
import numpy as np
import jax
import jax.random as random
from jax.scipy.stats.uniform import logpdf
from tqdm import tqdm
# from jax.experimental import host_callback as hcb


class MetropolisHastings:
    def __init__(self,p0,n_walkers,n_dim,num_burn,num_samples,next_pos_func=None,param_ranges=None):
        self.p0= p0
        self.n_walkers = n_walkers
        self.n_dim = n_dim
        self.num_burn = num_burn
        self.num_samples = num_samples
        self.next_pos_func = next_pos_func # Should not take in only random number key as argument
        self.param_ranges = param_ranges
        if self.next_pos_func is None:
            if self.param_ranges is None:
                raise ValueError("No function mentioned for getting next position from  current position.\
                    Please mention param_ranges if you want to use uniform continuos function for choosing next position.")
            else:
                pass
        
        # The uniform case is default implemented with param_ranges. It should be like an array of size (n_dim,2).
        # Note that func arguments should be like func(p0 or p_next,args)
        
    @partial(jax.jit, static_argnums=(0,2,))
    def next_position_walkers(self,prngKey,log_func,position,prob_pos):
        k1,k2 = jax.random.split(prngKey,2)
        next_pos = None
        if self.next_pos_func is None:
            if self.param_ranges is not None:
                min_val,max_val = self.param_ranges.T
                next_pos = random.uniform(k2,shape=(self.n_walkers,self.n_dim,),minval=jnp.array(min_val),maxval=jnp.array(max_val))
        else:
            next_pos = self.next_pos_func()
        
        
        log_current = log_func(position)
        log_next = log_func(next_pos)
        x = random.uniform(k1,shape=log_current.shape)
        log_unif = jnp.log(x)  # Log of a uniform sample

        
        move = log_next > log_current + log_unif
        move = move[:,np.newaxis]
        
        position = jnp.where(move,next_pos,position)
        
        return position,prob_pos
    
    def run_mcmc(self,log_func):
        def step(carry,key):
            position,prob_pos = carry
            new_position,new_prob = self.next_position_walkers(key,log_func,position,prob_pos)
            return (new_position,new_prob),new_position
        keys = random.split(random.PRNGKey(100),self.num_samples+self.num_burn)
        (_,__),samples = jax.lax.scan(step,(self.p0,log_func(self.p0)),keys)
        
        return samples
    
    
