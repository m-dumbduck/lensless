from datasets import load_dataset
from huggingface_hub import snapshot_download

REPO_ID = "bezzam/DigiCam-Mirflickr-MultiMask-10K"

ds = load_dataset("bezzam/DigiCam-Mirflickr-MultiMask-10K")
ds.save_to_disk("./data/DigiCam-Mirflickr-MultiMask-10K/dataset/")
snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    local_dir="./data/DigiCam-Mirflickr-MultiMask-10K",
    allow_patterns=["masks/**"],
)
