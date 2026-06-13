import torch
from torch import nn


class Permute(nn.Module):
    def __init__(self, data_object_key: str, order):
        super().__init__()
        self.data_object_key = data_object_key
        self.order = tuple(order)

    def forward(self, x):
        """
        Args:
            x (Tensor): input tensor.
        Returns:
            x (Tensor): permuted tensor.
        """
        x[self.data_object_key] = x[self.data_object_key].permute(self.order)
        return x
