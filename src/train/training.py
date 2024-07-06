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
from src.util.tiling import (
    image_to_patches,
    patches_to_image_weighted_average,
    patches_to_image,
)

pl.Config.set_tbl_rows(1000)
amp_enabled = True


class Trainer:
    def __init__(
        self,
        model,
        device,
        train_dataset,
        optimizer: torch.optim.Optimizer,
        output_dir,
        outer_patch_size,
        inner_patch_size,
        val_dataset=None,
        batch_size=1,
        output_name="modulated_siren",
        save_interval=100,
    ):
        self.model: nn.Module = model.to(device)
        self.device = device
        self.outer_patch_size = outer_patch_size
        self.inner_patch_size = inner_patch_size
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
        )
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
            )
        else:
            self.val_loader = None
        self.optimizer = optimizer
        self.output_name = output_name
        self.output_dir = output_dir
        self.criterion = nn.MSELoss()
        self.writer = SummaryWriter(log_dir=f"{output_dir}/{output_name}/runs")
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
            save_interval,
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)

    def train(self, num_epochs):
        for _ in range(num_epochs):
            training_loss = self._train_epoch()
            validation_loss = self._validate_epoch() if self.val_loader else 0
            self.training_manager.post_epoch_update(training_loss, validation_loss)
        self.training_manager.update_human_readable_short_progress_log()
        self.training_manager.save_model("final")

    def _train_epoch(self):
        self.model.train()
        training_loss = 0
        for fully_sampled_batch, undersampled_batch in self.train_loader:
            training_loss += self._train_iteration(
                undersampled_batch, fully_sampled_batch
            )
        return training_loss

    def _train_iteration(self, undersampled_batch, fully_sampled_batch):
        undersampled_batch = undersampled_batch.to(self.device).float()

        fully_sampled_batch = (
            extract_center_batch(
                fully_sampled_batch, self.outer_patch_size, self.inner_patch_size
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
        self.model.eval()
        validation_loss = 0
        with torch.no_grad():
            for fully_sampled_batch, undersampled_batch in self.val_loader:
                validation_loss += self._validate_iteration(
                    undersampled_batch, fully_sampled_batch
                )
        return validation_loss

    def _validate_iteration(self, undersampled_batch, fully_sampled_batch):
        undersampled_batch = undersampled_batch.to(self.device).float()
        fully_sampled_batch = (
            extract_center_batch(
                fully_sampled_batch, self.outer_patch_size, self.inner_patch_size
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
        with torch.no_grad():
            training_loss = 0
            validation_loss = 0
            for fully_sampled_batch, undersampled_batch in self.train_loader:
                fully_sampled_batch = (
                    extract_center_batch(
                        fully_sampled_batch,
                        self.outer_patch_size,
                        self.inner_patch_size,
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
                            self.inner_patch_size,
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
        save_interval=100,
    ):
        self.model = model
        self.optimizer = optimizer
        self.output_dir = output_dir
        self.output_name = output_name
        self.train_loader = train_loader
        self.train_dataset = train_dataset
        self.val_loader = val_loader
        self.val_dataset = val_dataset
        self.device = device
        self.save_interval = save_interval
        self.epoch_counter = 0
        self.batch_counter = 0
        self.starting_time = time.time()
        self.progress_log: pl.DataFrame = pl.from_dict(
            {
                "epoch": [-1],
                "training_loss": [initial_training_loss],
                "validation_loss": [initial_validation_loss],
                "time_since_start": [0.0],
            }
        )

    def post_epoch_update(self, training_loss, validation_loss):
        if self.epoch_counter % self.save_interval == 0:
            self.save_model()
            self.save_snapshot()
        if self.epoch_counter % 100 == 0:
            self.update_human_readable_short_progress_log()
        self.epoch_counter += 1
        self.update_progress_log(training_loss, validation_loss)
        # Perform other post-epoch updates here

    def post_batch_update(self, loss):
        self.batch_counter += 1
        # Perform post-batch updates here

    def save_model(self, suffix=""):
        if not os.path.exists(f"{self.output_dir}/{self.output_name}/models"):
            os.makedirs(f"{self.output_dir}/{self.output_name}/models")
        torch.save(
            self.model.state_dict(),
            f"{self.output_dir}/{self.output_name}/models/{self.output_name}_model_epoch_{self.epoch_counter}_{suffix}.pth",
        )
        torch.save(
            self.optimizer.state_dict(),
            f"{self.output_dir}/{self.output_name}/models/{self.output_name}_optimizer_epoch_{self.epoch_counter}_{suffix}.pth",
        )

    def save_snapshot(self):
        output_dir = f"{self.output_dir}/{self.output_name}/snapshots"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i in 0, 1:
            fully_sampled, undersampled = self.train_dataset.get_random_image()
            fully_sampled = fully_sampled.unsqueeze(0).to(self.device).float()
            undersampled = undersampled.unsqueeze(0).to(self.device).float()

            fully_sampled, fully_sampled_information = image_to_patches(
                fully_sampled, 32, 16
            )
            undersampled, undersampled_information = image_to_patches(
                undersampled, 32, 16
            )

            with torch.no_grad():
                output = self.model(undersampled)

            save_image_comparison(
                patches_to_image(fully_sampled, fully_sampled_information, 32, 16),
                patches_to_image(undersampled, undersampled_information, 32, 16),
                patches_to_image_weighted_average(
                    output, undersampled_information, 16, 16
                ),
                f"{output_dir}/snapshot_train_epoch_{self.epoch_counter}_{i}.png",
            )

            if self.val_dataset:
                fully_sampled, undersampled = self.val_dataset.get_random_image()
                fully_sampled = fully_sampled.unsqueeze(0).to(self.device).float()
                undersampled = undersampled.unsqueeze(0).to(self.device).float()

                fully_sampled, fully_sampled_information = image_to_patches(
                    fully_sampled, 32, 16
                )
                undersampled, undersampled_information = image_to_patches(
                    undersampled, 32, 16
                )

                with torch.no_grad():
                    output = self.model(undersampled)

                save_image_comparison(
                    patches_to_image(fully_sampled, fully_sampled_information, 32, 16),
                    patches_to_image(undersampled, undersampled_information, 32, 16),
                    patches_to_image_weighted_average(
                        output, undersampled_information, 16, 16
                    ),
                    f"{output_dir}/snapshot_val_epoch_{self.epoch_counter}_{i}.png",
                )

    def update_progress_log(self, training_loss, validation_loss):
        current_time = time.time()
        current_log = {
            "epoch": self.epoch_counter,
            "training_loss": training_loss,
            "validation_loss": validation_loss,
            "time_since_start": float(
                round((current_time - self.starting_time) / 60, 0)
            ),
        }
        self.progress_log = self.progress_log.extend(pl.from_dict(current_log))

    def update_human_readable_short_progress_log(self):
        """Create .txt file with human readable short progress log.
        Short meaning only every n-th epoch is included.
        """
        short_log = self.progress_log.gather_every(20)
        last_log = self.progress_log[-1, :]
        short_log = short_log.extend(last_log)
        short_log = short_log.unique(maintain_order=True, subset=["epoch"])
        with open(f"{self.output_dir}/progress_log.txt", "w", encoding="utf-8") as f:
            print(short_log, file=f)
