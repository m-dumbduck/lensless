import io

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def make_comparison_figure(lensless, reconstructed, lensed):
    lensless = lensless.permute(1, 2, 0).detach().cpu().numpy()
    reconstructed = reconstructed.permute(1, 2, 0).detach().cpu().numpy()
    lensed = lensed.permute(1, 2, 0).detach().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(lensless)
    axes[0].set_title("Lensless")
    axes[0].axis("off")

    axes[1].imshow(reconstructed)
    axes[1].set_title("Reconstructed")
    axes[1].axis("off")

    axes[2].imshow(lensed)
    axes[2].set_title("Lensed")
    axes[2].axis("off")

    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))
