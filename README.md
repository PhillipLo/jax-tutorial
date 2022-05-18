# jax-tutorial
A tutorial on JAX.

# Conda environment stuff
Before doing anything, run `conda env create -f environment.yml`.

# Files
`README.md`: That's me! \
`.gitignore`: gitignore file. \
`environment.yml`: Conda environment for this project. \
`tutorial.ipynb`: Self-contained Jupyter Notebook containing lots of JAX demos; cells must be run in order! \
`fwd_rev_comparison.py`: Experiment demonstrating when to use foward vs reverse mode differentiation for functions with Jacobians of different sizes. To run: `python fwd_rev_comparison.py`, will save results to `fwd_rev_comparison.png`. \
`jit_comparison.py`: Experiment demonstrating performance boost when jitting. To run: `python jit_comparison.py`, will save results in `jit_comparison.png`. \
