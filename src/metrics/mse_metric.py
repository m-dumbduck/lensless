import torch

from src.metrics.base_metric import BaseMetric


class MSEMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, reconstructed: torch.Tensor, lensed: torch.Tensor, **kwargs):
        B = reconstructed.shape[0]
        mse_metric = ((lensed - reconstructed).reshape(B, -1) ** 2).mean(dim=1).mean()
        return mse_metric.detach().cpu().item()
