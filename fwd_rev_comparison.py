# run on CPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
from jax import jacfwd, jacrev
import jax.numpy as jnp
import time

import matplotlib.pyplot as plt


def f(x):
  '''
  A function mapping an x in R^100 to something in R.

  f(x) = \|(e^x - x^2) / tan(x)\|_2
  '''
  return jnp.linalg.norm((jnp.exp(x) - x**2) / jnp.tan(x))


def g(t, n):
  '''
  A parametric curve mapping a t in R to something in R^n, n an integer

  g(t) = [1, t, t^2, ..., t^n]
  '''
  return t ** jnp.sin(jnp.linspace(0, n - 1, n))

def main():
  f_jacfwd_times = []
  f_jacrev_times = []

  g_jacfwd_times = []
  g_jacrev_times = []

  ns = range(500, 10000, 500)

  for n in ns:
    x = jnp.linspace(0, 1, n)
    t = 0.5

    t0 = time.time()
    np.array(jacfwd(f)(x)) # convert to numpy in case JAX does something weird with async dispatch
    t1 = time.time()
    f_jacfwd_times.append(t1 - t0)

    t0 = time.time()
    np.array(jacrev(f)(x))
    t1 = time.time()
    f_jacrev_times.append(t1 - t0)

    t0 = time.time()
    np.array(jacfwd(g, argnums = 0)(t, n))
    t1 = time.time()
    g_jacfwd_times.append(t1 - t0)

    t0 = time.time()
    np.array(jacrev(g, argnums = 0)(t, n))
    t1 = time.time()
    g_jacrev_times.append(t1 - t0)

  plt.plot(ns, f_jacfwd_times, "b", label = "jacfwd(f) times")
  plt.plot(ns, f_jacrev_times, "c", label = "jacrev(f) times")
  plt.plot(ns, g_jacfwd_times, "r", label = "jacfwd(g) times")
  plt.plot(ns, g_jacrev_times, "m", label = "jacrev(g) times")

  plt.legend()
  plt.xlabel("n")
  plt.ylabel("compute time")
  plt.title("Forward and reverse mode differentiation performance comparison")
  plt.tight_layout()
  plt.savefig("fwd_rev_comparison", dpi = 100)
  

if __name__=="__main__":
  main()

  

  



