import torch
from torch import nn


class LPIPSLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, lensed: torch.Tensor, restored: torch.Tensor, **batch):
        return {"loss": self.loss(lensed, restored)}
