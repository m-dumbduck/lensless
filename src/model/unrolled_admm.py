from torch import nn
import torch
from src.model.base_model.base_model import BaseModel
from src.model.optimizers.admm_optimizer import ADMMOptimizer
import numpy as np


class UnrolledADMM(BaseModel):
    def __init__(self, n_iters: int, mu1_init: float, mu2_init: float, mu3_init: float, tau_init: float):
        super().__init__()
        self.register_buffer("n_iters", torch.tensor(n_iters))
        self.log_mu1 = nn.Parameter(torch.full((n_iters,), np.log(mu1_init)))
        self.log_mu2 = nn.Parameter(torch.full((n_iters,), np.log(mu2_init)))
        self.log_mu3 = nn.Parameter(torch.full((n_iters,), np.log(mu3_init)))
        self.log_tau = nn.Parameter(torch.full((n_iters,), np.log(tau_init)))

    def forward(self, lensless, psf, **batch):
        optimizer = ADMMOptimizer(b=lensless, H=psf)
        for step in range(self.n_iters):
            optimizer.step(
                self.log_mu1[step].exp(), self.log_mu2[step].exp(),
                self.log_mu3[step].exp(), self.log_tau[step].exp()
            )
        return {"reconstructed": optimizer.get_result()}
