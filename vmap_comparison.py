import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from jax import jit, random, vmap

import numpy as np
import jax.numpy as jnp

import time

import matplotlib.pyplot as plt

def frob_norm(A):
  '''
  Compute the Frobenus norm of a matrix A
  '''
  return jnp.linalg.norm(A)

def manual_batched_frob_norm(As):
  n = As.shape[0]
  norms = jnp.zeros(shape = As.shape[0])
  for i in range(n):
    norms = norms.at[i].set(frob_norm(As[i]))
  return norms

vmap_batched_frob_norm = vmap(frob_norm)

jitted_vmap_batched_frob_norm = jit(vmap_batched_frob_norm)


def main():
  key, subkey = random.split(random.PRNGKey(0))

  manual_batched_times = []
  vmap_batched_times = []
  jitted_vmap_batched_times = []
  
  n = 1000

  for _ in range(100):
    As = random.normal(subkey, shape = (100, n, n))
    key, subkey = random.split(key)

    t0 = time.time()
    np.array(manual_batched_frob_norm(As))
    t1 = time.time()
    manual_batched_times.append(t1 - t0)

    t0 = time.time()
    np.array(vmap_batched_frob_norm(As))
    t1 = time.time()
    vmap_batched_times.append(t1 - t0)

    t0 = time.time()
    np.array(jitted_vmap_batched_frob_norm(As))
    t1 = time.time()
    jitted_vmap_batched_times.append(t1 - t0)

  # discard first 10 runs to account for compilation burn-in time
  np_avg = np.mean(manual_batched_times[10:])
  jnp_avg = np.mean(vmap_batched_times[10:])
  jnp_jitted_avg = np.mean(jitted_vmap_batched_times[10:])

  methods = ["manual batched", "vmap batched", "jit vmap batched"]
  times = np.array([np_avg, jnp_avg, jnp_jitted_avg])

  plt.bar(methods, times)

  plt.xlabel("method", fontdict = {"fontsize": 16})
  plt.ylabel("mean compute time in s",  fontdict = {"fontsize": 16})
  plt.title("vmap performance, discarding first 10 runs",  fontdict = {"fontsize": 16})
  plt.tight_layout()
  plt.savefig("vmap_comparison", dpi = 100)


if __name__=="__main__":
  main()

