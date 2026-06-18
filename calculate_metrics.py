import argparse
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from PIL import Image

from src.lensless_helpers.preprocessor import (
    ALIGNMENT,
    convert_image_to_float,
    force_rgb,
    get_cropped_lensed,
    get_roi,
)

from src.metrics.tracker import MetricTracker


def load_image(path):
    return convert_image_to_float(force_rgb(np.array(Image.open(path))))


def to_chw(image):
    return torch.from_numpy(image).permute(2, 0, 1)


def align_ground_truth(image):
    canvas_height = ALIGNMENT["top_left"][0] + ALIGNMENT["height"]
    canvas_width = ALIGNMENT["top_left"][1] + ALIGNMENT["width"]
    reference = np.zeros((canvas_height, canvas_width, 3))
    return get_roi(get_cropped_lensed(image, reference))


def build_metrics():
    with initialize(version_base=None, config_path="src/configs/metrics"):
        config = compose(config_name="base")
    return instantiate(config.inference)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True, help="Dir with ground truth images")
    parser.add_argument("--pred", required=True, help="Dir with reconstructed images")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    gt_dir = Path(args.gt)
    pred_dir = Path(args.pred)

    metrics = build_metrics()
    tracker = MetricTracker(*[metric.name for metric in metrics])

    matched = 0
    for pred_path in sorted(pred_dir.glob("*.png")):
        gt_path = gt_dir / pred_path.name
        if not gt_path.exists():
            continue
        matched += 1

        reconstructed = to_chw(load_image(pred_path)).unsqueeze(0).to(args.device)
        lensed = to_chw(align_ground_truth(load_image(gt_path))).unsqueeze(0).to(args.device)

        for metric in metrics:
            tracker.update(metric.name, metric(reconstructed=reconstructed, lensed=lensed))

    print(f"Metrics over {matched} image(s):")
    for name, value in tracker.result().items():
        print(f"    {name}: {value}")


if __name__ == "__main__":
    main()
