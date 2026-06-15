import torch
import lpips
from torch import nn


class LPIPSLoss(nn.Module):
    def __init__(self, net, normalize=False):
        super().__init__()
        self.lpips_model = lpips.LPIPS(net=net)
        self.normalize = normalize

    def forward(self, lensed: torch.Tensor, reconstructed: torch.Tensor, **batch):
        return {"loss": self.lpips_model(reconstructed, lensed, normalize=self.normalize).mean()}
