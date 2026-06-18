import torch
from tqdm.auto import tqdm
from torchvision.utils import save_image
import time

from src.lensless_helpers.preprocessor import get_roi
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from src.utils.drawer_utils import make_comparison_figure


class Inferencer(BaseTrainer):
    """
    Inferencer (Like Trainer but for Inference) class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.
    """

    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        save_path,
        metrics=None,
        writer=None,
        logger=None,
        batch_transforms=None,
        skip_model_load=False,
        pretrained_type=None
    ):
        """
        Initialize the Inferencer.

        Args:
            model (nn.Module): PyTorch model.
            config (DictConfig): run config containing inferencer config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            save_path (str): path to save model predictions and other
                information.
            metrics (dict): dict with the definition of metrics for
                inference (metrics[inference]). Each metric is an instance
                of src.metrics.BaseMetric.
            batch_transforms (dict[nn.Module] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
            skip_model_load (bool): if False, require the user to set
                pre-trained checkpoint path. Set this argument to True if
                the model desirable weights are defined outside of the
                Inferencer Class.
        """
        assert (
            skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config.inferencer

        self.pretrained_type = pretrained_type

        self.device = device

        self.model = model
        self.batch_transforms = batch_transforms

        # define dataloaders
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}

        # logging
        self.logger = logger
        self.writer = writer
        self.log_step = config.inferencer.get("log_step", 50)

        # path definition

        self.save_path = save_path

        # define metrics
        self.metrics = metrics
        if self.metrics is not None:
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in self.metrics["inference"]],
                writer=self.writer,
            )
        else:
            self.evaluation_metrics = None

        if not skip_model_load:
            # init model
            self._from_pretrained(config.inferencer.get("from_pretrained"))

    def run_inference(self):
        """
        Run inference on each partition.

        Returns:
            part_logs (dict): part_logs[part_name] contains logs
                for the part_name partition.
        """
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs
        return part_logs

    def process_batch(self, batch_idx, batch, metrics, part):
        """
        Run batch through the model, compute metrics, and
        save predictions to disk.

        Save directory is defined by save_path in the inference
        config and current partition.

        Args:
            batch_idx (int): the index of the current batch.
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type
                of the partition (train or inference).
            part (str): name of the partition. Used to define proper saving
                directory.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform)
                and model outputs.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        if not self.cfg_trainer.get("measure_speed", False):
            outputs = self.model(**batch)
        else:
            outputs = self._forward_model(batch, batch_idx)

        batch.update(outputs)

        batch["reconstructed"] = get_roi(batch["reconstructed"])
        has_lensed = "lensed" in batch
        if has_lensed:
            batch["lensed"] = get_roi(batch["lensed"])

        if metrics is not None and has_lensed:
            for met in self.metrics["inference"]:
                value = met(**batch)
                if value is None:
                    continue
                metrics.update(met.name, value)

        if self.save_path is not None:
            n = batch["reconstructed"].shape[0]
            save_dir = self.save_path / part
            save_dir.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                img_id = batch["id"][i] if "id" in batch.keys() else batch_idx * n + i
                save_image(batch["reconstructed"][i], save_dir / f"{img_id}.png")

        return batch

    def _sync_if_needed(self):
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _forward_model(self, batch, batch_idx):
        n_warmup = self.cfg_trainer.get("speed_num_warmup", 5)
        runs_per_batch = self.cfg_trainer.get("speed_num_repeat", 1)

        if batch_idx < n_warmup:
            return self.model(**batch)

        batch_size = batch["lensless"].shape[0]
        times = []
        outputs = {}
        for _ in range(runs_per_batch):
            self._sync_if_needed()
            start = time.perf_counter()
            outputs = self.model(**batch)
            self._sync_if_needed()
            elapsed = time.perf_counter() - start
            times.append(1000 * elapsed / batch_size)

        outputs["batch_time"] = sum(times) / len(times)
        return outputs

    def _log_batch(self, batch_idx, batch, mode):
        if self.writer is None or "lensed" not in batch:
            return

        lensless = batch["lensless"][0]
        reconstructed = batch["reconstructed"][0]
        lensed = batch["lensed"][0]

        self.writer.add_image(
            f"{mode}/comparison",
            make_comparison_figure(lensless, reconstructed, lensed),
        )

    def _inference_part(self, part, dataloader):
        """
        Run inference on a given partition and save predictions

        Args:
            part (str): name of the partition.
            dataloader (DataLoader): dataloader for the given partition.
        Returns:
            logs (dict): metrics, calculated on the partition.
        """

        self.is_train = False
        self.model.eval()

        self.evaluation_metrics.reset()

        # create Save dir
        if self.save_path is not None:
            (self.save_path / part).mkdir(exist_ok=True, parents=True)

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    part=part,
                    metrics=self.evaluation_metrics,
                )
                if self.writer is not None and batch_idx % self.log_step == 0:
                    self.writer.set_step(batch_idx, part)
                    self._log_batch(batch_idx, batch, part)

        if self.writer is not None and self.evaluation_metrics is not None:
            self.writer.set_step(len(dataloader), part)
            self._log_scalars(self.evaluation_metrics)

        return self.evaluation_metrics.result()
