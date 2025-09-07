#!/bin/bash

CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/download_diffusion_renderer_checkpoints.py --checkpoint_dir checkpoints