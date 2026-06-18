import argparse

import torch
from hydra import compose, initialize
from hydra.utils import instantiate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--repo-id", required=True, help="HF repo id")
    parser.add_argument("--model-config", required=True, help="Model config name in src/configs/model")
    args = parser.parse_args()

    with initialize(version_base=None, config_path="../src/configs/model"):
        model_cfg = compose(config_name=args.model_config)

    model = instantiate(model_cfg)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint.get("state_dict", checkpoint))
    model.eval()

    model.push_to_hub(repo_id=args.repo_id)


if __name__ == "__main__":
    main()
