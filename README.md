
<div align="center">

  <img src="assets/deepseekjax.png" alt="A jax logo style image of a whale." width="200" height="auto" />
  <h1>Torch2Jax-DeepSeek-R1-Distill-Qwen-1.5B</h1>
  
  <p>
    Flax (JAX) implementation of DeepSeek-R1-Distill-Qwen-1.5B with weights ported from Hugging Face.
  </p>
  
  
<!-- Badges -->
<p>
  <a href="https://github.com/J-Rosser-UK/Torch2Jax-DeepSeek-R1-Distill-Qwen-1.5B/contributors">
    <img src="https://img.shields.io/github/contributors/J-Rosser-UK/Torch2Jax-DeepSeek-R1-Distill-Qwen-1.5B" alt="contributors" />
  </a>
  <a href="">
    <img src="https://img.shields.io/github/last-commit/J-Rosser-UK/Torch2Jax-DeepSeek-R1-Distill-Qwen-1.5B" alt="last update" />
  </a>
  <a href="https://github.com/J-Rosser-UK/Torch2Jax-DeepSeek-R1-Distill-Qwen-1.5B/network/members">
    <img src="https://img.shields.io/github/forks/J-Rosser-UK/Torch2Jax-DeepSeek-R1-Distill-Qwen-1.5B" alt="forks" />
  </a>
  <a href="https://github.com/J-Rosser-UK/Torch2Jax-DeepSeek-R1-Distill-Qwen-1.5B/stargazers">
    <img src="https://img.shields.io/github/stars/J-Rosser-UK/Torch2Jax-DeepSeek-R1-Distill-Qwen-1.5B" alt="stars" />
  </a>
  <a href="https://github.com/J-Rosser-UK/Torch2Jax-DeepSeek-R1-Distill-Qwen-1.5B/issues/">
    <img src="https://img.shields.io/github/issues/J-Rosser-UK/Torch2Jax-DeepSeek-R1-Distill-Qwen-1.5B" alt="open issues" />
  </a>
  <a href="https://github.com/J-Rosser-UK/Torch2Jax-DeepSeek-R1-Distill-Qwen-1.5B/blob/master/LICENSE">
    <img src="https://img.shields.io/github/license/J-Rosser-UK/Torch2Jax-DeepSeek-R1-Distill-Qwen-1.5B.svg" alt="license" />
  </a>
</p>
   
<h4>
    <a href="https://github.com/J-Rosser-UK/Torch2Jax-DeepSeek-R1-Distill-Qwen-1.5B/issues/">Report Bug</a>
  <span> · </span>
    <a href="https://github.com/J-Rosser-UK/Torch2Jax-DeepSeek-R1-Distill-Qwen-1.5B/issues/">Request Feature</a>
    <span> · </span>
    <a href="https://colab.research.google.com/drive/1jJaAARwbsFeV5hZoffrNwFhc8i2I-7ji?usp=sharing">Colab</a>
  </h4>
</div>



## Overview

Colab: https://colab.research.google.com/drive/1jJaAARwbsFeV5hZoffrNwFhc8i2I-7ji?usp=sharing

This repository provides both Flax (JAX) and PyTorch implementations of the DeepSeek-R1-Distill-Qwen-1.5B model. It includes:

- **Inference [QUICKSTART]**:
    - `inference.ipynb`: Contains a quickstart script to download and convert params from torch to flax, load model and perform text generation.

- **Flax Implementations**:  
    - `model_flax.py`: The Flax implementation.  

- **PyTorch Implementation**:  
    - `model_torch.py`: A reference implementation in PyTorch.

- **Conversion Script**:  
    - `torch_to_flax.py`: A utility to convert a PyTorch checkpoint (state dictionary) into Flax parameters.

## System Requirements
### Single GPU
16GB VRAM on the GPU + 64GB RAM (this can be swap)

### Multi-Device
Runs sharded on v2-8 TPU on Google Colab. 