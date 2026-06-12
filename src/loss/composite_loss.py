import torch
from torch import nn

from src.loss.lpips_loss import LPIPSLoss
from src.loss.mse_loss import MSELoss


class CompositeLoss(nn.Module):
    def __init__(
        self,
        mse_loss: MSELoss,
        mse_loss_weight: float,
        lpips_loss: LPIPSLoss,
        lpips_loss_weight: float,
    ):
        super().__init__()
        self.mse_loss = mse_loss
        self.mse_loss_weight = mse_loss_weight
        self.lpips_loss = lpips_loss
        self.lpips_loss_weight = lpips_loss_weight

    def forward(self, **batch):
        mse_loss = self.mse_loss(**batch)["loss"]
        lpips_loss = self.lpips_loss(**batch)["loss"]
        loss = self.mse_loss_weight * mse_loss + self.lpips_loss * lpips_loss
        return {
            "loss": loss,
            "mse_loss": mse_loss,
            "lpips_loss": lpips_loss
        }
