import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {}

    result_batch["lensless"] = torch.stack(
        [elem["lensless"] for elem in dataset_items], dim=0
    )
    result_batch["psf"] = torch.cat(
        [elem["psf"] for elem in dataset_items], dim=0
    )

    if "lensed" in dataset_items[0]:
        result_batch["lensed"] = torch.stack(
            [elem["lensed"] for elem in dataset_items], dim=0
        )

    if "id" in dataset_items[0]:
        result_batch["id"] = [elem["id"] for elem in dataset_items]

    return result_batch
