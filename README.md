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
python run.py experiment=ddpm_cifar10
```

## Models
- [DDPM](#ddpm)
- [NCSN](#ncsn)
- [NCSNV2](#ncsnv2)
- [DDIM](#ddim)
- [Score SDE](#score-sde)

---

### DDPM
_Denoising Diffusion Probabilistic Model_

#### Authors
Jonathan Ho, Ajay Jain, Pieter Abbeel

#### Run Example
```bash
python run.py experiment=ddpm_cifar10
```

---

### NCSN
_Generative Modeling by Estimating Gradients of the Data Distribution_

#### Authors
Yang Song, Stefano Ermon

#### Run Example
```bash
python run.py experiment=ncsn_cifar10
```
---

### NCSNV2
_Improved Techniques for Training Score-Based Generative Models_

#### Authors
Yang Song, Stefano Ermon

#### Run Example
```bash
python run.py experiment=ncsnv2_cifar10
```

---

### DDIM
_Denoising Diffusion Implicit Model_

#### Authors
Jiaming Song, Chenlin Meng, Stefano Ermon

#### Run Example
```bash
python run.py experiment=ddim_cifar10
```

---

### Score SDE
_Score-Based Generative Modeling through Stochastic Differential Equations_

#### Authors
Yang Song, Jashcha Sohl-Dickstein, Diederik P.Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole

#### Run Example
```bash
python run.py experiment=score_sde_cifar10
```