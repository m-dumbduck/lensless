import argparse
from pathlib import Path

import datasets
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--split", default="test")
    parser.add_argument("--dataset-path", default="data/DigiCam-Mirflickr-MultiMask-10K/dataset")
    parser.add_argument("--masks-path", default="data/DigiCam-Mirflickr-MultiMask-10K/masks")
    parser.add_argument("--out", default="data/example")
    args = parser.parse_args()

    ds = datasets.load_from_disk(args.dataset_path)[args.split]
    masks_path = Path(args.masks_path)

    out = Path(args.out)
    (out / "lensless").mkdir(parents=True, exist_ok=True)
    (out / "lensed").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)

    n = min(args.n, len(ds))
    for i in range(n):
        rec = ds[i]
        image_id = f"{i:04d}"

        rec["lensless"].save(out / "lensless" / f"{image_id}.png")
        rec["lensed"].save(out / "lensed" / f"{image_id}.png")

        mask_vals = np.load(masks_path / f"mask_{rec['mask_label']}.npy")
        np.save(out / "masks" / image_id, mask_vals)

        print(f"saved {image_id} (mask_label={rec['mask_label']})")

    print(f"\nDone. {n} samples -> {out}")
    print("Structure: lensless/<id>.png, masks/<id>.npy, lensed/<id>.png")


if __name__ == "__main__":
    main()
