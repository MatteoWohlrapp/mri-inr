"""
Training abstraction for the SIREN model.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
import os
from src.util.tiling import extract_center_batch
import os
import polars as pl
from src.data.mri_dataset import MRIDataset
from src.util.visualization import save_image_comparison
import numpy as np
from src.util.losses import PerceptualLoss, EdgeLoss
import pathlib
from src.util.tiling import (
    image_to_patches,
    patches_to_image_weighted_average,
    patches_to_image,
    filter_and_remember_black_patches,
    reintegrate_black_patches,
)

pl.Config.set_tbl_rows(1000)
amp_enabled = True


class Trainer:
    """
    Trainer for the SIREN model.
    """

    def __init__(
        self,
        model,
        device,
        train_dataset,
        optimizer: torch.optim.Optimizer,
        output_dir,
        outer_patch_size,
        inner_patch_size,
        siren_patch_size,
        val_dataset=None,
        batch_size=1,
        output_name="modulated_siren",
        num_workers=4,
        save_interval=100,
        logging=False,
        criterion="MSE",
    ):
        """
        Init

        Args:
            model (nn.Module): The model to train.
            device (torch.device): The device to use for training.
            train_dataset (MRIDataset): The dataset to use for training.
            optimizer (torch.optim.Optimizer): The optimizer to use for training.
            output_dir (str): The output directory.
            outer_patch_size (int): The size of the outer patch.
            inner_patch_size (int): The size of the inner patch.
            siren_patch_size (int): The size of the SIREN patch.
            val_dataset (MRIDataset, optional): The dataset to use for validation. Defaults to None.
            batch_size (int, optional): The batch size. Defaults to 1.
            output_name (str, optional): The output name. Defaults to "modulated_siren".
            num_workers (int, optional): The number of workers. Defaults to 4.
            save_interval (int, optional): The save interval. Defaults to 100.
            logging (bool, optional): Whether to log. Defaults
            criterion (str, optional): The criterion to use. Defaults to "MSE".
        """
        self.model: nn.Module = model.to(device)
        self.device = device
        self.outer_patch_size = outer_patch_size
        self.inner_patch_size = inner_patch_size
        self.siren_patch_size = siren_patch_size
        self.num_workers = num_workers
        self.criterion = criterion
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
            )
        else:
            self.val_loader = None
        self.optimizer = optimizer
        self.output_name = output_name
        self.output_dir = output_dir
        if self.criterion == "MSE":
            self.criterion = nn.MSELoss()
        elif self.criterion == "Perceptual":
            self.criterion = PerceptualLoss(
                pathlib.Path(
                    r"./output/perceptual_encoder/model_694.pth"
                ),
                nn.MSELoss(),
                self.device,
            )
        elif self.criterion == "Edge":
            self.criterion = EdgeLoss(
                nn.MSELoss(),
                self.device,
            )
        else:
            raise ValueError("Unsupported criterion")
        self.writer = (
            SummaryWriter(log_dir=f"{output_dir}/{output_name}/tensorboard")
            if logging
            else None
        )
        initial_training_loss, initial_validation_loss = self.get_initial_errors()
        self.training_manager = TrainingManager(
            model,
            optimizer,
            output_dir,
            output_name,
            self.train_loader,
            train_dataset,
            self.val_loader,
            val_dataset,
            initial_training_loss,
            initial_validation_loss,
            device,
            outer_patch_size,
            inner_patch_size,
            siren_patch_size,
            batch_size,
            save_interval,
            self.writer,
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)

    def train(self, initial_epoch, num_epochs):
        """
        Training loop.

        Args:
            initial_epoch (int): The initial epoch.
            num_epochs (int): The number of epochs.
        """
        self.training_manager.set_epoch_counter(initial_epoch)
        for _ in range(initial_epoch, num_epochs):
            training_loss = self._train_epoch()
            validation_loss = self._validate_epoch() if self.val_loader else 0
            self.training_manager.post_epoch_update(training_loss, validation_loss)
        self.training_manager.update_human_readable_short_progress_log()
        self.training_manager.save_model("final")
        if self.writer:
            self.writer.close()

    def _train_epoch(self):
        """
        Training epoch.

        Returns:
            float: The training loss.
        """
        self.model.train()
        training_loss = 0
        batches = 0
        for fully_sampled_batch, undersampled_batch in self.train_loader:
            batches += 1
            training_loss += self._train_iteration(
                undersampled_batch, fully_sampled_batch
            )
        return training_loss / batches

    def _train_iteration(self, undersampled_batch, fully_sampled_batch):
        """
        Training iteration.

        Args:
            undersampled_batch (torch.Tensor): The undersampled batch.
            fully_sampled_batch (torch.Tensor): The fully sampled batch.

        Returns:
            float: The loss.
        """
        undersampled_batch = undersampled_batch.to(self.device).float()

        fully_sampled_batch = (
            extract_center_batch(
                fully_sampled_batch, self.outer_patch_size, self.siren_patch_size
            )
            .to(self.device)
            .float()
        )
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            self.optimizer.zero_grad()
            outputs = self.model(undersampled_batch)
            loss = self.criterion(outputs, fully_sampled_batch)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            loss_item = loss.item()

        self.training_manager.post_batch_update(loss_item)
        return loss_item

    def _validate_epoch(self):
        """
        Validation epoch.

        Returns:
            float: The validation loss.
        """
        self.model.eval()
        validation_loss = 0
        batches = 0
        with torch.no_grad():
            for fully_sampled_batch, undersampled_batch in self.val_loader:
                batches += 1
                validation_loss += self._validate_iteration(
                    undersampled_batch, fully_sampled_batch
                )
        return validation_loss / batches

    def _validate_iteration(self, undersampled_batch, fully_sampled_batch):
        """
        Validation iteration.

        Args:
            undersampled_batch (torch.Tensor): The undersampled batch.
            fully_sampled_batch (torch.Tensor): The fully sampled batch.

        Returns:
            float: The loss.
        """
        undersampled_batch = undersampled_batch.to(self.device).float()
        fully_sampled_batch = (
            extract_center_batch(
                fully_sampled_batch, self.outer_patch_size, self.siren_patch_size
            )
            .to(self.device)
            .float()
        )
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            outputs = self.model(undersampled_batch)
            loss = self.criterion(outputs, fully_sampled_batch)

        return loss.item()

    def load_model(self, model_path, optimizer_path=None):
        """Used to continue training from a checkpoint."""
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        if optimizer_path:
            self.optimizer.load_state_dict(torch.load(optimizer_path))

    def get_initial_errors(self):
        """Get the initial training and validation errors."""
        with torch.no_grad():
            training_loss = 0
            validation_loss = 0
            for fully_sampled_batch, undersampled_batch in self.train_loader:
                fully_sampled_batch = (
                    extract_center_batch(
                        fully_sampled_batch,
                        self.outer_patch_size,
                        self.siren_patch_size,
                    )
                    .to(self.device)
                    .float()
                )
                outputs = self.model(undersampled_batch.to(self.device).float())
                training_loss += self.criterion(outputs, fully_sampled_batch).item()
            if self.val_loader:
                for fully_sampled_batch, undersampled_batch in self.val_loader:
                    fully_sampled_batch = (
                        extract_center_batch(
                            fully_sampled_batch,
                            self.outer_patch_size,
                            self.siren_patch_size,
                        )
                        .to(self.device)
                        .float()
                    )
                    outputs = self.model(undersampled_batch.to(self.device).float())
                    validation_loss += self.criterion(
                        outputs, fully_sampled_batch
                    ).item()
        return training_loss, validation_loss


class TrainingManager:
    """
    Manager handling training functionality
    """

    def __init__(
        self,
        model,
        optimizer,
        output_dir,
        output_name,
        train_loader,
        train_dataset,
        val_loader,
        val_dataset,
        initial_training_loss,
        initial_validation_loss,
        device,
        outer_patch_size,
        inner_patch_size,
        siren_patch_size,
        batch_size=1,
        save_interval=100,
        writer=None,
    ):
        """
        Init

        Args:
            model (nn.Module): The model to train.
            optimizer (torch.optim.Optimizer): The optimizer to use for training.
            output_dir (str): The output directory.
            output_name (str): The output name.
            train_loader (DataLoader): The training data loader.
            train_dataset (MRIDataset): The training dataset.
            val_loader (DataLoader): The validation data loader.
            val_dataset (MRIDataset): The validation dataset.
            initial_training_loss (float): The initial training loss.
            initial_validation_loss (float): The initial validation loss.
            device (torch.device): The device to use for training.
            outer_patch_size (int): The size of the outer patch.
            inner_patch_size (int): The size of the inner patch.
            siren_patch_size (int): The size of the SIREN patch.
            batch_size (int, optional): The batch size. Defaults to 1.
            save_interval (int, optional): The save interval. Defaults to 100.
            writer (SummaryWriter, optional): The tensorboard writer. Defaults to None.
        """
        self.model = model
        self.optimizer = optimizer
        self.output_dir = output_dir
        self.output_name = output_name
        self.train_loader = train_loader
        self.train_dataset = train_dataset
        self.val_loader = val_loader
        self.val_dataset = val_dataset
        self.device = device
        self.outer_patch_size = outer_patch_size
        self.inner_patch_size = inner_patch_size
        self.siren_patch_size = siren_patch_size
        self.save_interval = save_interval
        self.batch_size = batch_size
        self.writer = writer
        self.epoch_counter = 0
        self.batch_counter = 0
        self.starting_time = time.time()
        self.progress_log: pl.DataFrame = pl.from_dict(
            {
                "epoch": [0],
                "training_loss": [float(initial_training_loss)],
                "validation_loss": [float(initial_validation_loss)],
                "time_since_start": [float(0.0)],
            }
        )

    def post_epoch_update(self, training_loss, validation_loss):
        """
        Update after each epoch

        Args:
            training_loss (float): The training loss.
            validation_loss (float): The validation loss
        """
        if self.epoch_counter % self.save_interval == 0:
            self.save_model()
            self.save_snapshot()
        if self.epoch_counter % 100 == 0:
            self.update_human_readable_short_progress_log()
        self.epoch_counter += 1
        self.update_progress_log(training_loss, validation_loss)
        if self.writer:
            self.writer.add_scalar("training_loss", training_loss, self.epoch_counter)
            self.writer.add_scalar(
                "validation_loss", validation_loss, self.epoch_counter
            )

    def post_batch_update(self, loss):
        """
        Update after each batch

        Args:
            loss (float): The loss.
        """
        self.batch_counter += 1

    def save_model(self, suffix=""):
        """
        Save the model and optimizer.

        Args:
            suffix (str, optional): The suffix. Defaults to "".
        """
        if suffix != "":
            suffix = f"_{suffix}"
        if not os.path.exists(f"{self.output_dir}/{self.output_name}/models"):
            os.makedirs(f"{self.output_dir}/{self.output_name}/models")
        torch.save(
            self.model.state_dict(),
            f"{self.output_dir}/{self.output_name}/models/{self.output_name}_model_epoch_{self.epoch_counter}{suffix}.pth",
        )
        torch.save(
            self.optimizer.state_dict(),
            f"{self.output_dir}/{self.output_name}/models/{self.output_name}_optimizer_epoch_{self.epoch_counter}{suffix}.pth",
        )

    def save_snapshot(self):
        """
        Save a snapshot of the current training state.
        """
        output_dir = f"{self.output_dir}/{self.output_name}/snapshots"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i in 0, 1:
            fully_sampled, undersampled = self.train_dataset.get_random_image()
            fully_sampled = fully_sampled.unsqueeze(0).to(self.device).float()
            undersampled = undersampled.unsqueeze(0).to(self.device).float()

            fully_sampled, fully_sampled_information = image_to_patches(
                fully_sampled, self.outer_patch_size, self.inner_patch_size
            )
            undersampled, undersampled_information = image_to_patches(
                undersampled, self.outer_patch_size, self.inner_patch_size
            )

            with torch.no_grad():
                undersampled_filtered, filter_information, original_shape = (
                    filter_and_remember_black_patches(undersampled)
                )
                output = self.model(undersampled_filtered)
                output = reintegrate_black_patches(
                    output, filter_information, original_shape
                )

            save_image_comparison(
                patches_to_image(
                    fully_sampled,
                    fully_sampled_information,
                    self.outer_patch_size,
                    self.inner_patch_size,
                ),
                patches_to_image(
                    undersampled,
                    undersampled_information,
                    self.outer_patch_size,
                    self.inner_patch_size,
                ),
                patches_to_image_weighted_average(
                    output,
                    undersampled_information,
                    self.siren_patch_size,
                    self.inner_patch_size,
                    self.device,
                ),
                f"{output_dir}/snapshot_train_epoch_{self.epoch_counter}_{i}.png",
            )

            if self.val_dataset:
                fully_sampled, undersampled = self.val_dataset.get_random_image()
                fully_sampled = fully_sampled.unsqueeze(0).to(self.device).float()
                undersampled = undersampled.unsqueeze(0).to(self.device).float()

                fully_sampled, fully_sampled_information = image_to_patches(
                    fully_sampled, self.outer_patch_size, self.inner_patch_size
                )
                undersampled, undersampled_information = image_to_patches(
                    undersampled, self.outer_patch_size, self.inner_patch_size
                )

                with torch.no_grad():
                    undersampled_filtered, filter_information, original_shape = (
                        filter_and_remember_black_patches(undersampled)
                    )
                    output = self.model(undersampled_filtered)
                    output = reintegrate_black_patches(
                        output, filter_information, original_shape
                    )

                save_image_comparison(
                    patches_to_image(
                        fully_sampled,
                        fully_sampled_information,
                        self.outer_patch_size,
                        self.inner_patch_size,
                    ),
                    patches_to_image(
                        undersampled,
                        undersampled_information,
                        self.outer_patch_size,
                        self.inner_patch_size,
                    ),
                    patches_to_image_weighted_average(
                        output,
                        undersampled_information,
                        self.siren_patch_size,
                        self.inner_patch_size,
                        self.device,
                    ),
                    f"{output_dir}/snapshot_val_epoch_{self.epoch_counter}_{i}.png",
                )

    def update_progress_log(self, training_loss, validation_loss):
        """
        Update the progress log.

        Args:
            training_loss (float): The training loss.
            validation_loss (float): The validation loss.
        """
        current_time = time.time()
        current_log = {
            "epoch": self.epoch_counter,
            "training_loss": float(training_loss),
            "validation_loss": float(validation_loss),
            "time_since_start": float(
                round((current_time - self.starting_time) / 60, 0)
            ),
        }
        self.progress_log = self.progress_log.extend(pl.from_dict(current_log))

    def update_human_readable_short_progress_log(self):
        """
        Update the human readable short progress log.
        """
        output_dir = f"{self.output_dir}/{self.output_name}"
        os.makedirs(output_dir, exist_ok=True)
        """Create .txt file with human readable short progress log.
        Short meaning only every n-th epoch is included.
        """
        short_log = self.progress_log.gather_every(20)
        last_log = self.progress_log[-1, :]
        short_log = short_log.extend(last_log)
        short_log = short_log.unique(maintain_order=True, subset=["epoch"])
        with open(f"{output_dir}/progress_log.txt", "w", encoding="utf-8") as f:
            print(short_log, file=f)

    def set_epoch_counter(self, epoch):
        self.epoch_counter = epoch
