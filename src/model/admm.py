from torch import nn
import torch
from src.model.base_model import BaseModel
from src.model.admm_optimizer import ADMMOptimizer


class ADMM(BaseModel):
    def __init__(self, n_iters: int, mu1: float, mu2: float, mu3: float, tau: float):
        super().__init__()
        self.register_buffer("n_iters", torch.tensor(n_iters))
        self.register_buffer("mu1", torch.tensor(mu1))
        self.register_buffer("mu2", torch.tensor(mu2))
        self.register_buffer("mu3", torch.tensor(mu3))
        self.register_buffer("tau", torch.tensor(tau))

    def forward(self, lensless, psf, **batch):
        optimizer = ADMMOptimizer(
            b=lensless.permute(0, 3, 1, 2),
            H=psf.permute(0, 3, 1, 2),
            mu1=self.mu1, mu2=self.mu2, mu3=self.mu3, tau=self.tau
        )
        for _ in range(self.n_iters):
            optimizer.step()
        return {"reconstructed": optimizer.get_result().permute(0, 2, 3, 1)}
