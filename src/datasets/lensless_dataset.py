import numpy as np
import torch
from tqdm.auto import tqdm
from pathlib import Path

import datasets
from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class LenslessDataset(BaseDataset):
    """
    A dataset loaded from HuggingFace or local.
    """

    def __init__(
        self, dataset_path, masks_path, split="train", *args, **kwargs
    ):
        index = datasets.load_from_disk(dataset_path)[split]
        self.masks_path = masks_path
        super().__init__(index, *args, **kwargs)

    def get_mask(self, mask_index):
        return np.load(Path(self.masks_path) / f"mask_{mask_index}.npy")
