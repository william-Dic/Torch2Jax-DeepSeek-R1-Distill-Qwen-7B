# Torch2Jax-DeepSeek-R1-Distill-Qwen-1.5B

Flax (JAX) implementation of DeepSeek-R1-Distill-Qwen-1.5B with weights ported from Hugging Face.


## Overview

This repository provides both Flax (JAX) and PyTorch implementations of the DeepSeek-R1-Distill-Qwen-1.5B model. It includes:

- **Flax Implementations**:  
  - `model_flax.py`: A version with dynamic caching for efficient autoregressive generation.  
  - `model_flax_no_cache.py`: A variant without caching support.

- **PyTorch Implementation**:  
  - `model_torch.py`: A reference implementation in PyTorch.

- **Conversion Script**:  
  - `torch_to_flax.py`: A utility to convert a PyTorch checkpoint (state dictionary) into Flax parameters.
