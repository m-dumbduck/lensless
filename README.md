# Lensless Imaging

This repository contains implementations of ADMM-based reconstruction
algorithms for mask-based lensless imaging for the HSE DL Research homework 5.

We follow
[Learned reconstructions for practical mask-based lensless imaging](https://arxiv.org/abs/1908.11502)
and
[Towards Robust and Generalizable Lensless Imaging with Modular Learned Reconstruction](https://arxiv.org/abs/2502.01102).
All algorithms are implemented from scratch.

ADMM is computed with FFTs on a padded space and regularized with an
anisotropic TV term. The reconstruction is cropped out after the iterations.

We implement and compare the following methods:

- ADMM-100: classic ADMM, 100 iterations.
- Unrolled ADMM-20: 20 iterations.
- Modular LeADMM-5: 5 iterations with learned DRUNet processors, in three
  variants (`le_admm_pre_post`, `le_admm_pre`, `le_admm_post`).

ADMM-100 uses fixed hyperparameters ($\mu_i = 10^{-4}$, $\tau = 2\cdot10^{-4}$)
and is not trained. Unrolled ADMM-20 is the same except
per-iteration $\mu_1, \mu_2, \mu_3, \tau$ become trainable (parametrized in log-space).
The modular LeADMM-5 variants wrap a 5-iteration unrolled ADMM with learned
DRUNet processors, which are applied before or after the reconstruction (or both). 
All LeADMM-5 variants similarly to the [paper](https://arxiv.org/abs/2502.01102) 
are 8M-parameter.

Trainable models are optimized with a composite loss: `MSE`
(weight `1`) + `LPIPS` (VGG, weight `0.1`).

Quality is reported using metrics: `PSNR`, `LPIPS` (VGG), `MSE` and `SSIM`.


### Dataset

For training and evaluation we use
[DigiCam-Mirflickr-MultiMask-10K](https://huggingface.co/datasets/bezzam/DigiCam-Mirflickr-MultiMask-10K)
(`train` split for training, `test` split for evaluation).

Download the dataset and the per-sample masks with the provided script:

```bash
python scripts/download_hf_dataset.py
```

This saves the dataset to `data/DigiCam-Mirflickr-MultiMask-10K/dataset` and the
masks to `data/DigiCam-Mirflickr-MultiMask-10K/masks`.


### Training and Evaluation

We provide a [CometML report](https://www.comet.com/m-dumbduck/lensless/reports/eWfA0BxIXwDJCDySUkJFDdKpX/) with full training logs,
reconstructed images, and metric curves for all final models.

Trained models are published on HuggingFace:

- Unrolled ADMM-20: [ldiujes/unrolled-admm](https://huggingface.co/ldiujes/unrolled-admm)
- LeADMM-5 (pre): [ldiujes/le-admm-pre](https://huggingface.co/ldiujes/le-admm-pre)
- LeADMM-5 (post): [ldiujes/le-admm-post](https://huggingface.co/ldiujes/le-admm-post)
- LeADMM-5 (pre+post): [ldiujes/le-admm-pre-post](https://huggingface.co/ldiujes/le-admm-pre-post)

Final evaluation on `DigiCam-Mirflickr-MultiMask-10K test` (top model, LeADMM-5 with preprocessor and postprocessor):

- `PSNR  = 16.4`
- `LPIPS = 0.565`
- `MSE   = 0.024`
- `SSIM  = 0.441`


## Installation

```bash
git clone https://github.com/m-dumbduck/lensless.git
cd lensless

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```


## Demo notebook

We provide a fully standalone `demo.ipynb` notebook that demonstrates the
trained model. Given a custom URL to a `.zip` dataset on Google Drive (a default
URL is provided), it downloads the data, runs `inference.py`, saves
reconstructions, visualizes samples (original — lensless — reconstruction), and,
if ground-truth images are present, runs `calculate_metrics.py` to report the
four metrics. The required model checkpoints are pulled from HuggingFace
automatically.

A user only has to pass the link and run the cells — no other steps are
expected. Before launching the notebook you may want to create virtual
environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

There are two scenarios in which notebook can be used.

1. **Notebook is installed on its own.** In this case by running its cells the notebook clones
the repository to a folder near itself. You may want to
use this scenario if you want to inference our model in Colab/Kaggle.
2. **Notebook is installed with repository.** In this case
notebook will also work properly. Running its cells will not trigger any repository cloning.



## Training

Training utilizes Hydra configs in `src/configs`. By default `train.py` runs
the `le_admm_pre_post` model:

```bash
python train.py
```

To train other models, pass the corresponding config name:

```bash
python train.py --config-name unrolled_admm_train
python train.py --config-name le_admm_pre_train
python train.py --config-name le_admm_post_train
python train.py --config-name le_admm_pre_post_train
```

Each training config lets you adjust model hyperparameters, loss weights,
logging frequency, number of epochs, and epoch length. Use `trainer.resume_from`
to continue from a saved checkpoint. Losses, metrics, and images are logged to
CometML (using the `writer` subconfig).

> **Note:** ADMM-100 has no trainable parameters and is used at inference only.


## Inference and Evaluation

`inference.py` applies a model to a dataset and saves the reconstructions to
`data/saved/<save_path>` (each reconstruction id matches its input image id).
The default config loads the pretrained `le-admm-pre-post` model from
HuggingFace and runs on a `CustomDirDataset`:

```bash
python inference.py
```

When the dataset provides ground-truth images, `inference.py` also computes and
prints metrics `PSNR`, `LPIPS`, `MSE`, `SSIM`

### Reproducing the reported metrics

The numbers above come from running the trained models on the DigiCam test
split. For the top model (LeADMM-5 with pre- and post-processor):

```bash
python inference.py datasets=lensless_eval
```

For the fixed ADMM-100 baseline (no checkpoint needed):

```bash
python inference.py --config-name admm_inference
```

The pretrained source is controlled by `inferencer.from_pretrained` and
`inferencer.from_pretrained_type` (`hf` for a HuggingFace repo id, `file` for a
local checkpoint).

### Custom directory inference

`CustomDirDataset` (`src/configs/datasets/custom_dir.yaml`) parses any directory
of the form:

```
NameOfTheDirectoryWithData
├── lensless
│   └── <id>.png
├── masks
│   └── <id>.npy
└── lensed        # ground truth, optional
    └── <id>.png
```

Point the dataset at your directory and pick an output name:

```bash
python inference.py datasets.test.data_dir=<path-to-dir> inferencer.save_path=<name>
```

You can build such an example directory from the dataset with:

```bash
python scripts/make_example_dir.py --n 50 --out data/example
```

### Metrics

For custom directories (and in the demo), metrics are computed by a separate
script, given a ground-truth directory and a reconstruction directory (matched
by file name):

```bash
python calculate_metrics.py --gt <gt_dir> --pred <pred_dir>
```

It prints `PSNR`, `LPIPS`, `MSE`, and `SSIM`.

### Reconstruction speed

To measure per-image reconstruction speed (with warmup and repeats), use:

```bash
python inference.py --config-name inference_measure_speed
```


## Export to Hugging Face

A trained checkpoint can be exported to HuggingFace with `PyTorchModelHubMixin`:

```bash
python scripts/export_to_hub.py \
  --checkpoint saved/<run>/checkpoint-epochN.pth \
  --repo-id your_hf_nickname/your_hf_repo \
  --model-config le_admm_pre_post
```

After export, the model can be loaded directly:

```python
from src.model import LeADMM

model = LeADMM.from_pretrained("ldiujes/le-admm-pre-post")
```


## Credits

This repository is based on the pytorch project template
[Blinorot/pytorch_project_template](https://github.com/Blinorot/pytorch_project_template).


## Used links

- [Learned reconstructions for practical mask-based lensless imaging](https://arxiv.org/abs/1908.11502)
- [Towards Robust and Generalizable Lensless Imaging with Modular Learned Reconstruction](https://arxiv.org/abs/2502.01102)
- [DigiCam-Mirflickr-MultiMask-10K dataset](https://huggingface.co/datasets/bezzam/DigiCam-Mirflickr-MultiMask-10K)
- [GitHub repository](https://github.com/m-dumbduck/lensless)
- [CometML report](https://www.comet.com/m-dumbduck/lensless/reports/eWfA0BxIXwDJCDySUkJFDdKpX/)


## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
