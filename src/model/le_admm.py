from src.model.unrolled_admm import UnrolledADMM
from src.model.drunet.drunet import DRUNet


class LeADMM(UnrolledADMM):
    def __init__(self, n_iters: int, mu1_init: float, mu2_init: float, mu3_init: float, tau_init: float, preprocessor_channels: list[int] = None, postprocessor_channels: list[int] = None):
        super().__init__(n_iters, mu1_init, mu2_init, mu3_init, tau_init)
        self.preprocessor = None
        if preprocessor_channels is not None:
            self.preprocessor = DRUNet(3, 3, preprocessor_channels)
        self.postprocessor = None
        if postprocessor_channels is not None:
            self.postprocessor = DRUNet(3, 3, postprocessor_channels)

    def forward(self, lensless, psf, **batch):
        if self.preprocessor is not None:
            lensless = self.preprocessor(lensless)
        reconstructed = super().forward(lensless=lensless, psf=psf, **batch)["reconstructed"]
        if self.postprocessor is not None:
            reconstructed = self.postprocessor(reconstructed)
        return {"reconstructed": reconstructed}
