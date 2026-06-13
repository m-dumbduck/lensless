import torch
from src.metrics.base_metric import BaseMetric
from torchmetrics.image import StructuralSimilarityIndexMeasure


class StructuralSimilarityIndexMeasureMetric(BaseMetric):
    def __init__(self, data_range, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=data_range)

    def __call__(self, reconstructed: torch.Tensor, lensed: torch.Tensor, **kwargs):
        self.ssim_metric = self.ssim_metric.to(reconstructed.device)
        return self.ssim_metric(reconstructed, lensed).detach().cpu().item()
