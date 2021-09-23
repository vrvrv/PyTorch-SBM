<h1 align="center">
  <b>PyTorch SBM</b><br>
</h1>

<p align="center">
      <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/Python-3.8-blue.svg" /></a>
       <a href= "https://pytorch.org/">
        <img src="https://img.shields.io/badge/PyTorch-1.4.5-brightgreen.svg" /></a>
       <a href= "https://github.com/vrvrv/PyTorch-SBM/blob/master/LICENSE">
        <img src="https://img.shields.io/badge/license-Apache2.0-blue.svg" /></a>
</p>

This repository contains a collection of PyTorch-Lightning implementations of Score-based Generative Model presented in research papers.
Contributions and suggestions of SBMs to implement are very welcomed.

### Requirements
- Python >= 3.8
- PyTorch >= 1.9
- Pytorch Lightning >= 1.4.5
- CUDA enabled computing device

### Installation
```bash
git clone https://github.com/vrvrv/PyTorch-SBM
cd PyTorch-SBM
pip install -r requirements.txt
```

### How to run
```bash
python run experiment=ddpm_cifar10
```

### Models
- [DDPM](#DDPM)

### DDPM
_Denoising Diffusion Probabilistic Model_

#### Authors
Jonathan Ho, Ajay Jain, Pieter Abbeel

#### Run Example
```bash
python run.py experiment=ddpm_cifar10
```

