import torch
from src.metrics.base_metric import BaseMetric
from torchmetrics.image import PeakSignalNoiseRatio


class PeakSignalNoiseRatioMetric(BaseMetric):
    def __init__(self, data_range, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ssim_metric = PeakSignalNoiseRatio(data_range=data_range)

    def __call__(self, reconstructed: torch.Tensor, lensed: torch.Tensor, **kwargs):
        return self.ssim_metric(reconstructed, lensed).detach().cpu().item()
