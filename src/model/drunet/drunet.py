from torch import nn
from src.model.base_model.base_model import BaseModel
from src.model.drunet.residual_block_stack import ResidualBlockStack
import torch.nn.functional as F

class DRUNet(BaseModel):
    def __init__(self, in_channels: int, out_channels: int, internal_channels: list[int]):
        super().__init__()
        reversed_channels = internal_channels[::-1]
        self.in_conv = nn.Conv2d(in_channels=in_channels, out_channels=internal_channels[0], kernel_size=3, padding=1)
        self.downscaling_sequence = nn.ModuleList([
            nn.ModuleList([
                ResidualBlockStack(num_blocks=4, outer_channels=in_c, internal_channels=in_c),
                nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=2, stride=2, padding=0)
            ])
            for in_c, out_c in zip(internal_channels[:-1], internal_channels[1:])
        ])
        self.middle_conv = ResidualBlockStack(num_blocks=4, outer_channels=internal_channels[-1], internal_channels=internal_channels[-1])
        self.upscaling_sequence = nn.ModuleList([
            nn.ModuleList([
                nn.ConvTranspose2d(in_channels=in_c, out_channels=out_c, kernel_size=2, stride=2, padding=0),
                ResidualBlockStack(num_blocks=4, outer_channels=out_c, internal_channels=out_c)
            ])
            for in_c, out_c in zip(reversed_channels[:-1], reversed_channels[1:])
        ])
        self.out_conv = nn.Conv2d(in_channels=internal_channels[0], out_channels=out_channels, kernel_size=3, padding=1)
        self.factor = 2 ** len(self.downscaling_sequence)

    def forward(self, X):
        _, _, H, W = X.shape
        pad_h = (self.factor - H % self.factor) % self.factor
        pad_w = (self.factor - W % self.factor) % self.factor
        X = F.pad(X, (0, pad_w, 0, pad_h), mode="reflect")

        X = self.in_conv(X)
        X_0 = X
        residuals = []
        for res_block_stack, conv in self.downscaling_sequence:
            X = res_block_stack(X)
            X = conv(X)
            residuals.append(X)
        X = self.middle_conv(X)
        for (conv_t, res_block_stack), residual in zip(self.upscaling_sequence, reversed(residuals)):
            X = conv_t(X + residual)
            X = res_block_stack(X)
        X = self.out_conv(X + X_0)

        X = X[..., :H, :W]

        return X
