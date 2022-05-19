import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from jax import jit, random

import numpy as np
import jax.numpy as jnp

import time

import matplotlib.pyplot as plt


def render_np(P, w, sigma, d, num):
  '''
  Render the collection of Bezier curves 
  with control points P, weights w, and 
  blur sigma, using numpy
  INPUTS:
    P: array of control points for n cubic Bezier curves, shape (n, 4, 2), values in [0, d]
    w: array of weights for Bezier curves, shape (n,), values in [0, 1]
    sigma: float, value >= 1
    d: int, size of rendered image
    num: number of integration samples to take 

  OUTPUTS:
    rendering: (d, d, 1)-array, the rendered image
  '''
  ts = np.linspace(0, 1, num = num)

  xx, yy = np.meshgrid(np.arange(d), np.arange(d))
  coord_grid = np.stack((xx.T, yy.T), axis = 2)

  coeff = 1 / (2 * np.pi * sigma**2)

  # Compute the (num, 4)-shaped array where
  # T[k] is all polynomial operations needed
  # evaluated at t_k
  T = np.array([(1 - ts)**3,
                  3 * ((1 - ts)**2 * ts),
                  3 * (1 - ts) * ts**2,
                  ts**3]).T

  # Compute the (d, d, n, num, 2)-shaped array
  # where diff[a1, a2, i, k] is equal to
  # C_{p_i}(t_j) - [a1, a2] 
  diff = np.einsum("kj, ijl -> ikl", T, P) - coord_grid[:, :, None, None, :]

  # Compute the (d, d, n, num)-shaped array
  # that's the equivalent of np.linalg.norm(diff, axis = -1)**2
  diff_norm_sq = np.einsum('ijklm, ijklm -> ijkl', diff, diff)
  
  # Compute the (d, d, n, num)-shaped array
  # where f[a1, a2, i, k] is equal to f(p^i, sigma, t_k)
  f = np.exp(-diff_norm_sq / (2 * sigma**2))

  # Compute the (n, num, 2)-shaped array 
  # where path_deriv[i, k] = C_{p^i}'(t_k)
  path_deriv = (np.einsum('k,il->ikl', ts**2, (-3*P[:,0,:] + 9*P[:,1,:] - 9*P[:,2,:] + 3*P[:,3,:]))
    + np.einsum('k,il->ikl', ts,(6*P[:,0,:] - 12*P[:,1,:] + 6*P[:,2,:])) 
    + (-3*P[:,0,:] + 3*P[:,1,:])[:, None,:])

  # Compute the (n, num)-shaped array
  # where g[i, k] is equal to g(p^i,t_k)
  g = np.linalg.norm(path_deriv, axis = -1) 

  # Compute the (d, d, n, num)-shaped array 
  # where integrand[a1, a2, j, k] is equal to 
  # \frac{1}{2\pi\sigma^2}f(p^i,sigma,t_k,a)g(p^i,t_k)
  integrands = coeff * f * g 

  # Perform a trapezoidal integration along the last axis of integrands.
  integrals = (1/(2 * (num-1))) * (2 * (np.sum(integrands, axis = -1)) - integrands[:,:,:,0] - integrands[:,:,:,-1])

  rendering = np.einsum('ijk, k -> ij', integrals, w)
  rendering = np.expand_dims(rendering, axis = -1)

  return rendering


def render_jnp(P, w, sigma, d, num):
  '''
  Render the collection of Bezier curves 
  with control points P, weights w, and 
  blur sigma, using jax.numpy
  INPUTS:
    P: array of control points for n cubic Bezier curves, shape (n, 4, 3), values in [0, d]
    w: array of weights for Bezier curves, shape (n,), values in [0, 1]
    sigma: float, value >= 1
    d: int, size of rendered image
    num: number of integration samples to take 

  OUTPUTS:
    rendering: (d, d, 1)-array, the rendered image
  '''
  ts = jnp.linspace(0, 1, num = num)

  xx, yy = jnp.meshgrid(jnp.arange(d), jnp.arange(d))
  coord_grid = jnp.stack((xx.T, yy.T), axis = 2)

  coeff = 1 / (2 * jnp.pi * sigma**2)

  # Compute the (num, 4)-shaped array where
  # T[k] is all polynomial operations needed
  # evaluated at t_k
  T = jnp.array([(1 - ts)**3,
                  3 * ((1 - ts)**2 * ts),
                  3 * (1 - ts) * ts**2,
                  ts**3]).T

  # Compute the (d, d, n, num, 2)-shaped array
  # where diff[a1, a2, i, k] is equal to
  # C_{p_i}(t_j) - [a1, a2] 
  diff = jnp.einsum("kj, ijl -> ikl", T, P) - coord_grid[:, :, None, None, :]

  # Compute the (d, d, n, num)-shaped array
  # that's the equivalent of np.linalg.norm(diff, axis = -1)**2
  diff_norm_sq = jnp.einsum('ijklm, ijklm -> ijkl', diff, diff)
  
  # Compute the (d, d, n, num)-shaped array
  # where f[a1, a2, i, k] is equal to f(p^i, sigma, t_k)
  f = jnp.exp(-diff_norm_sq / (2 * sigma**2))

  # Compute the (n, num, 2)-shaped array 
  # where path_deriv[i, k] = C_{p^i}'(t_k)
  path_deriv = (jnp.einsum('k,il->ikl', ts**2, (-3*P[:,0,:] + 9*P[:,1,:] - 9*P[:,2,:] + 3*P[:,3,:]))
    + jnp.einsum('k,il->ikl', ts,(6*P[:,0,:] - 12*P[:,1,:] + 6*P[:,2,:])) 
    + (-3*P[:,0,:] + 3*P[:,1,:])[:, None,:])

  # Compute the (n, num)-shaped array
  # where g[i, k] is equal to g(p^i,t_k)
  g = jnp.linalg.norm(path_deriv, axis = -1) 

  # Compute the (d, d, n, num)-shaped array 
  # where integrand[a1, a2, j, k] is equal to 
  # \frac{1}{2\pi\sigma^2}f(p^i,sigma,t_k,a)g(p^i,t_k)
  integrands = coeff * f * g 

  # Perform a trapezoidal integration along the last axis of integrands.
  integrals = (1/(2 * (num-1))) * (2 * (jnp.sum(integrands, axis = -1)) - integrands[:,:,:,0] - integrands[:,:,:,-1])

  rendering = jnp.einsum('ijk, k -> ij', integrals, w)
  rendering = jnp.expand_dims(rendering, axis = -1)

  return rendering


def main():
  key, subkey = random.split(random.PRNGKey(0))

  np_times = []
  jnp_times = []
  jnp_jitted_times = []
  
  d = 64
  num = 50
  n = 8

  render_jnp_jitted = jit(render_jnp, static_argnums = (3, 4))

  for _ in range(200):
    P = random.uniform(subkey, shape = (n, 4, 2))
    key, subkey = random.split(key)

    w = random.uniform(subkey, shape = (n,))
    key, subkey = random.split(key)

    sigma = random.uniform(subkey, minval = 1, maxval = 3, shape = ())
    key, subkey = random.split(key)

    t0 = time.time()
    np.array(render_np(P, w, sigma, d, num))
    t1 = time.time()
    np_times.append(t1 - t0)

    t0 = time.time()
    np.array(render_jnp(P, w, sigma, d, num))
    t1 = time.time()
    jnp_times.append(t1 - t0)

    t0 = time.time()
    np.array(render_jnp_jitted(P, w, sigma, d, num))
    t1 = time.time()
    jnp_jitted_times.append(t1 - t0)

  np_avg = np.mean(np_times[10:])
  jnp_avg = np.mean(jnp_times[10:])
  jnp_jitted_avg = np.mean(jnp_jitted_times[10:])

  methods = ["numpy", "jnp unjitted", "jnp jitted"]
  times = np.array([np_avg, jnp_avg, jnp_jitted_avg])

  plt.bar(methods, times)

  plt.xlabel("method", fontdict = {"fontsize": 16})
  plt.ylabel("mean compute time in s",  fontdict = {"fontsize": 16})
  plt.title("Rendering performance, discarding first 10 runs",  fontdict = {"fontsize": 16})
  plt.tight_layout()
  plt.savefig("render_jit_comparison", dpi = 100)


if __name__=="__main__":
  main()