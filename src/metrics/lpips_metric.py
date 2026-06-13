import torch
import lpips
from src.metrics.base_metric import BaseMetric


class LPIPSMetric(BaseMetric):
    def __init__(self, net, normalize=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lpips_model = lpips.LPIPS(net=net)
        self.normalize = normalize

    def __call__(self, reconstructed: torch.Tensor, lensed: torch.Tensor, **kwargs):
        self.lpips_model = self.lpips_model.to(reconstructed.device)
        return self.lpips_model(reconstructed, lensed, normalize=self.normalize).detach().cpu().item()
