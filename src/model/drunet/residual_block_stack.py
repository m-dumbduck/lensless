from torch import nn

from src.model.base_model.base_model import BaseModel
from src.model.drunet.residual_block import ResidualBlock


class ResidualBlockStack(BaseModel):
    def __init__(self, num_blocks: int, outer_channels: int, internal_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            *[
                ResidualBlock(outer_channels=outer_channels, internal_channels=internal_channels)
                for _ in range(num_blocks)
            ]
        )

    def forward(self, X):
        return self.net(X)
