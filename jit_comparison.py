# run on CPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import jax.numpy as jnp
from jax import jit, random

import time

import matplotlib.pyplot as plt


def f_np(Ihat, I):
  '''
  Take the sigmoid of the squared l2 loss between two n x n images Ihat and I, using np functions.
  '''
  diff = np.linalg.norm(Ihat - I) **2 / 2
  return 1 / (1 + np.exp(-diff))


def f_jnp(Ihat, I):
  '''
  Take the sigmoid of the squared l2 loss between two n x n images Ihat and I, using jnp functions.
  '''
  diff = jnp.linalg.norm(Ihat - I) **2 / 2
  return 1 / (1 + jnp.exp(-diff))


def main():
  key, subkey = random.split(random.PRNGKey(0))

  np_times = []
  jnp_times = []
  jnp_jitted_times = []
  
  n = 128

  f_jnp_jitted = jit(f_jnp) # jitting is so easy!

  for _ in range(1000):
    Ihat = random.normal(subkey, shape = (n, n))
    key, subkey = random.split(key)

    I = random.normal(subkey, shape = (n, n))
    key, subkey = random.split(key)

    t0 = time.time()
    np.array(f_np(Ihat, I))
    t1 = time.time()
    np_times.append(t1 - t0)

    t0 = time.time()
    np.array(f_jnp(Ihat, I))
    t1 = time.time()
    jnp_times.append(t1 - t0)

    t0 = time.time()
    np.array(f_jnp_jitted(Ihat, I))
    t1 = time.time()
    jnp_jitted_times.append(t1 - t0)

  np_avg = np.mean(np_times[10:])
  jnp_avg = np.mean(jnp_times[10:])
  jnp_jitted_avg = np.mean(jnp_jitted_times[10:])

  methods = ["numpy", "jnp unjitted", "jnp jitted"]
  times = np.array([np_avg, jnp_avg, jnp_jitted_avg])

  plt.bar(methods, times * 1e6)

  plt.xlabel("method", fontdict = {"fontsize": 16})
  plt.ylabel("mean compute time in μs",  fontdict = {"fontsize": 16})
  plt.title("Jitting performance, discarding first 10 runs",  fontdict = {"fontsize": 16})
  plt.tight_layout()
  plt.savefig("jit_comparison", dpi = 100)


if __name__=="__main__":
  main()