# MatterRL
De novo material design using diffusion generation and reinforcement learning

# Installation

- We'll use `conda` to install dependencies and set up the environment for a Nvidia GPU machine.
- After installing `conda`, install [`mamba`](https://mamba.readthedocs.io/en/latest/) to the base environment. `mamba` is a faster, drop-in replacement for `conda`:
    ```bash
    conda install mamba -n base -c conda-forge
    ```
- Then create a conda environment and install the dependencies for diffusion models and RL:
    ```bash
    mamba env create -f env.yml
    ```
    Activate the conda environment with `conda activate matrl`.
