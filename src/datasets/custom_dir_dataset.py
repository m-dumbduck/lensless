from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image

from src.datasets.base_dataset import BaseDataset
from src.lensless_helpers.preprocessor import convert_image_to_float, force_rgb, get_cropped_lensed
from src.lensless_helpers.psf import simulate_psf_from_mask


class CustomDirDataset(BaseDataset):
    def __init__(self, data_dir, has_lensed=None, *args, **kwargs):
        data_dir = Path(data_dir)
        self.lensless_dir = data_dir / "lensless"
        self.masks_dir = data_dir / "masks"
        self.lensed_dir = data_dir / "lensed"

        if has_lensed is None:
            has_lensed = self.lensed_dir.is_dir() and any(self.lensed_dir.glob("*.png"))
        self.has_lensed = has_lensed

        index = []
        for lensless_path in sorted(self.lensless_dir.glob("*.png")):
            image_id = lensless_path.stem
            index.append({
                "lensless": str(lensless_path),
                "mask": str(self.masks_dir / f"{image_id}.npy"),
                "lensed": str(self.lensed_dir / f"{image_id}.png") if has_lensed else None,
                "id": image_id
            })
        super().__init__(index, *args, **kwargs)

    def __getitem__(self, ind):
        item = self._index[ind]

        lensless = convert_image_to_float(
            force_rgb(np.array(Image.open(item["lensless"])))
        )
        lensless = torch.rot90(torch.from_numpy(lensless), dims=(-3, -2), k=2)

        psf = simulate_psf_from_mask(np.load(item["mask"]))

        instance_data = {"id": item["id"], "lensless": lensless, "psf": psf}

        if item["lensed"] is not None:
            lensed = convert_image_to_float(
                force_rgb(np.array(Image.open(item["lensed"])))
            )
            instance_data["lensed"] = torch.from_numpy(
                get_cropped_lensed(lensed, lensless)
            )

        return self.preprocess_data(instance_data)

    def get_mask(self, mask_index):
        return np.load(self.masks_dir / f"{mask_index}.npy")

    @staticmethod
    def _assert_index_is_valid(index):
        for entry in index:
            assert "lensless" in entry and "mask" in entry, (
                "Each CustomDirDataset item must contain 'lensless' and 'mask' paths."
            )

    @staticmethod
    def _shuffle_and_limit_index(index, limit, shuffle_index):
        if shuffle_index:
            random.seed(42)
            random.shuffle(index)
        if limit is not None:
            index = index[:limit]
        return index
