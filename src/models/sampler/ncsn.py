import torch
import torch.nn as nn


class NCSNSampler(nn.Module):
    def __init__(self, score, sigmas, alphas, n_steps_each: int):
        super().__init__()
        self.score = score
        self.sigmas = sigmas
        self.alphas = alphas
        self.n_steps_each = n_steps_each

    def forward(self, x_T: torch.Tensor):
        for i, sigma_i in enumerate(self.sigmas):
            labels = (i * torch.ones_like(x_T[0])).long()
            for t in range(self.n_steps_each):
                x_T = x_T + self.alphas[i] * self.score(x_T, labels) / 2 + self.alphas[i] * torch.randn_like(x_T)

        return x_T

