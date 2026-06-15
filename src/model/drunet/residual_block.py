from torch import nn

from src.model.base_model.base_model import BaseModel


class ResidualBlock(BaseModel):
    def __init__(self, outer_channels: int, internal_channels: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=outer_channels,
                out_channels=internal_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=internal_channels,
                out_channels=outer_channels,
                kernel_size=3,
                padding=1,
            ),
        )

    def forward(self, X):
        return X + self.net(X)
