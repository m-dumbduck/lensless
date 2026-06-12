import torch
from torch import nn


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, lensed: torch.Tensor, reconstructed: torch.Tensor, **batch):
        return {"loss": self.loss(lensed, reconstructed)}
