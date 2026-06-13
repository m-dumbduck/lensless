import torch
from src.metrics.base_metric import BaseMetric
from torchmetrics.image import PeakSignalNoiseRatio


class PeakSignalNoiseRatioMetric(BaseMetric):
    def __init__(self, data_range, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.psnr_metric = PeakSignalNoiseRatio(data_range=data_range)

    def __call__(self, reconstructed: torch.Tensor, lensed: torch.Tensor, **kwargs):
        self.psnr_metric = self.psnr_metric.to(reconstructed.device)
        return self.psnr_metric(reconstructed, lensed).detach().cpu().item()
