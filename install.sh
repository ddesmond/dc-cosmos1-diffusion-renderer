#!/bin/bash

cd /opt

git clone https://github.com/nv-tlabs/cosmos1-diffusion-renderer.git

cd /opt/cosmos1-diffusion-renderer

# Create the cosmos-predict1 conda environment.
conda env create --file cosmos-predict1.yaml
# Activate the cosmos-predict1 conda environment.
conda activate cosmos-predict1
# Install the dependencies.
pip install -r requirements.txt
# Patch Transformer engine linking issues in conda environments.
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/python3.10
# Install Transformer engine.
pip install transformer-engine[pytorch]==1.12.0

# Patch dependency for nvdiffrast
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/triton/backends/nvidia/include/crt $CONDA_PREFIX/include/
pip install git+https://github.com/NVlabs/nvdiffrast.git

echo "Done installing cosmos1-diffusion-renderer"