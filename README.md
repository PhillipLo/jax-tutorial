# jax-tutorial
A tutorial on [JAX](https://jax.readthedocs.io/en/latest/).

# Conda environment stuff
Before doing anything, run `conda env create -f environment.yml`. *Note that this version of JAX breaks on M1 Mac devices.* It also doesn't come with GPU support; see the [installation guide](https://github.com/google/jax#installation) for GPU installation instructions.

# Files
`README.md`: That's me! 

`.gitignore`: gitignore file.

`environment.yml`: Conda environment for this project.

`tutorial.ipynb`: Self-contained Jupyter Notebook containing lots of JAX demos; cells must be run in order!

`fwd_rev_comparison.py`: Experiment demonstrating when to use foward vs reverse mode differentiation for functions with Jacobians of different sizes. To run: `python fwd_rev_comparison.py`, will save results to `fwd_rev_comparison.png`.

`jit_comparison.py`: Experiment demonstrating performance boost when jitting a simple function. To run: `python jit_comparison.py`, will save results in `jit_comparison.png`.

`render_comparison.py`: Experiment demonstrating performance boost when jitting Bezier render function. To run: `python render_comparison.py`, will save results in `render_jit_comparison.png`.

`vmap_comparison.py`: Experiment demonstrating performance boost when vmapping the computation of the Frobenius norm of batches of matrices. To run: `vmap_comparison.py`, will save rresults in `vmap_comparison.png`.
