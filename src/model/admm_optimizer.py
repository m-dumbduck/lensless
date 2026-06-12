import torch

class ADMMOptimizer:
    def __init__(self, b: torch.Tensor, H: torch.Tensor, mu1: float, mu2: float, mu3: float, tau: float, padding_scale: int=2):
        B, C, height, width = b.shape
        self.height = height
        self.width = width
        self.b = self._center_pad(b, padding_scale * b.shape[2], padding_scale * b.shape[3])
        self.v = torch.zeros_like(self.b)
        self.u = torch.zeros_like(self.b)
        self.u = torch.stack([self.u, self.u], dim=1)
        self.w = torch.zeros_like(self.b)
        self.x = torch.zeros_like(self.b)
        self.alpha1 = torch.zeros_like(self.b)
        self.alpha2 = torch.zeros_like(self.b)
        self.alpha2 = torch.stack([self.alpha2, self.alpha2], dim=1)
        self.alpha3 = torch.zeros_like(self.b)
        self.H = self._center_pad(H, padding_scale * H.shape[2], padding_scale * H.shape[3])
        self.H = torch.fft.ifftshift(self.H, dim=(-2, -1))
        self.H_fft = torch.fft.fft2(self.H)
        self.mu1 = mu1
        self.mu2 = mu2
        self.mu3 = mu3
        self.tau = tau
        delta = torch.zeros(1, 1, *self.b.shape[-2:], device=self.b.device)
        delta[..., 0, 0] = 1
        Dx_fft = torch.fft.fft2(self._Dx(delta))
        Dy_fft = torch.fft.fft2(self._Dy(delta))
        PhiT_Phi_fft = torch.abs(Dx_fft) ** 2 + torch.abs(Dy_fft) ** 2
        self.lst_sq_denom_fft = (
            self.mu1 * torch.abs(self.H_fft) ** 2 +
            self.mu2 * torch.abs(PhiT_Phi_fft) +
            self.mu3
        )

    def step(self):
        self.u = self._soft_thresholding(
            self._Phi(self.x) + self.alpha2 / self.mu2
        )
        self.v = (
            self.alpha1 + self.mu1 * self._H(self.x) + self.b
        ) / (1 + self.mu1)
        self.w = torch.clamp(self.alpha3 / self.mu3 + self.x, min=0)
        R = (
            self.mu3 * self.w - self.alpha3 +
            self._PhiT(self.mu2 * self.u - self.alpha2) +
            self._HT(self.mu1 * self.v - self.alpha1)
        )
        self.x = torch.fft.ifft2(torch.fft.fft2(R) / self.lst_sq_denom_fft).real
        self.alpha1 = self.alpha1 + self.mu1 * (self._H(self.x) - self.v)
        self.alpha2 = self.alpha2 + self.mu2 * (self._Phi(self.x) - self.u)
        self.alpha3 = self.alpha3 + self.mu3 * (self.x - self.w)

    def get_result(self):
        return self.x[..., self.bottom:self.bottom + self.height, self.left:self.left + self.width]

    def _H(self, X):
        return torch.fft.ifft2(self.H_fft * torch.fft.fft2(X)).real

    def _HT(self, X):
        return torch.fft.ifft2(self.H_fft.conj() * torch.fft.fft2(X)).real

    def _Dx(self, X):
        return torch.roll(X, shifts=-1, dims=-1) - X

    def _Dy(self, X):
        return torch.roll(X, shifts=-1, dims=-2) - X

    def _DxT(self, X):
        return torch.roll(X, shifts=1, dims=-1) - X

    def _DyT(self, X):
        return torch.roll(X, shifts=1, dims=-2) - X

    def _Phi(self, X):
        return torch.stack((self._Dx(X), self._Dy(X)), dim=1)

    def _PhiT(self, X):
        return self._DxT(X[:, 0]) + self._DyT(X[:, 1])

    def _soft_thresholding(self, X):
        return torch.sign(X) * torch.clamp(abs(X) - self.tau / self.mu2, min=0)

    def _center_pad(self, X, height, width):
        B, C, H, W = X.shape
        newX = torch.zeros(B, C, height, width, device=X.device)
        self.bottom = (height - H) // 2
        self.left = (width - W) // 2
        newX[..., self.bottom:self.bottom + H, self.left:self.left + W] = X
        return newX
