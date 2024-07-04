import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import time
import pathlib
import os
from src.util.tiling import collate_fn, extract_center_batch
import os
import polars as pl
from src.data.mri_dataset import MRIDataset
from src.util.visualization import save_image_comparison

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
        output_name="output",
        limit_io=False,
    ):
        self.model: nn.Module = model.to(device)
        self.device = device
        self.outer_patch_size = outer_patch_size
        self.inner_patch_size = inner_patch_size
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4
        )
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4
            )
        else:
            self.val_loader = None
        self.optimizer = optimizer
        self.output_name = output_name
        self.output_dir = output_dir
        self.output_model_dir = pathlib.Path(output_dir) / 'mod_siren' / f'mod_siren_{time.strftime("%Y-%m-%d_%H-%M-%S")}'
        self.output_model_dir.mkdir(parents=False, exist_ok=False)
        self.limit_io = limit_io
        self.criterion = nn.MSELoss()
        self.writer = SummaryWriter(log_dir=f"{output_dir}/runs/{output_name}")
        self.training_manager = TrainingManager(model, optimizer, self.output_model_dir)
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)

    def train(self, num_epochs):
        for _ in range(num_epochs):
            training_loss = self._train_epoch()
            validation_loss = self._validate_epoch() if self.val_loader else 0
            self.training_manager.post_epoch_update(training_loss, validation_loss)
        self.training_manager.update_human_readable_short_progress_log()
        self._save_model()

    def _train_epoch(self):
        self.model.train()
        training_loss = 0
        for (fully_sampled_batch, undersampled_batch) in self.train_loader:
            training_loss += self._train_iteration(undersampled_batch, fully_sampled_batch)
        return training_loss

    def _train_iteration(self, undersampled_batch, fully_sampled_batch):
        undersampled_batch = undersampled_batch.to(self.device).float()
        fully_sampled_batch = extract_center_batch(fully_sampled_batch,self.outer_patch_size, self.inner_patch_size).to(self.device).float()
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
            for (fully_sampled_batch, undersampled_batch) in self.val_loader:
                validation_loss += self._validate_iteration(undersampled_batch, fully_sampled_batch)
        return validation_loss

    def _validate_iteration(self, undersampled_batch, fully_sampled_batch):
        undersampled_batch = undersampled_batch.to(self.device).float()
        fully_sampled_batch = extract_center_batch(fully_sampled_batch,self.outer_patch_size, self.inner_patch_size).to(self.device).float()
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            outputs = self.model(undersampled_batch)
            loss = self.criterion(outputs, fully_sampled_batch)

        return loss.item()

    def _save_model(self):
        with torch.no_grad():
            if not os.path.exists(f"{self.output_model_dir}/model_checkpoints"):
                os.makedirs(f"{self.output_model_dir}/model_checkpoints")

            torch.save(
                self.model.state_dict(),
                f"{self.output_model_dir}/model_checkpoints/{self.output_name}_model.pth",
            )
            torch.save(
                self.optimizer.state_dict(),
                f"{self.output_model_dir}/model_checkpoints/{self.output_name}_optimizer.pth",
            )
    def load_model(self, model_path, optimizer_path=None):
        """Used to continue training from a checkpoint."""
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        if optimizer_path:
            self.optimizer.load_state_dict(torch.load(optimizer_path))
        

class TrainingManager:
    def __init__(self, model, optimizer, output_dir, train_loader, val_loader, save_interval=100):
        self.model = model
        self.optimizer = optimizer
        self.output_dir = output_dir
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_interval = save_interval
        self.epoch_counter = 0
        self.batch_counter = 0
        self.starting_time = time.time()
        self.progress_log: pl.DataFrame = pl.from_dict(
            {
                "epoch": [-1],
                "training_loss": [0.0],
                "validation_loss": [0.0],
                "time_since_start": [0.0],
            }
        )

    def post_epoch_update(self, training_loss, validation_loss):
        if self.epoch_counter % self.save_interval == 0:
            self.save_model()
        if self.epoch_counter % 100 == 0:
            self.update_human_readable_short_progress_log()
        self.epoch_counter += 1
        self.update_progress_log(training_loss, validation_loss)
        # Perform other post-epoch updates here

    def post_batch_update(self, loss):
        self.batch_counter += 1
        # Perform post-batch updates here

    def save_model(self):
        if not os.path.exists(f"{self.output_dir}/model_checkpoints"):
            os.makedirs(f"{self.output_dir}/model_checkpoints")
        torch.save(
            self.model.state_dict(),
            f"{self.output_dir}/model_checkpoints/model_epoch_{self.epoch_counter}.pth",
        )
        torch.save(
            self.optimizer.state_dict(),
            f"{self.output_dir}/model_checkpoints/optimizer_epoch_{self.epoch_counter}.pth",
        )

    def update_progress_log(self, training_loss, validation_loss):
        current_time = time.time()
        current_log = {
            "epoch": self.epoch_counter,
            "training_loss": training_loss,
            "validation_loss": validation_loss,
            "time_since_start": int(round((current_time - self.starting_time) / 60,0)),
        }
        self.progress_log = self.progress_log.extend(pl.from_dict(current_log))
    
    def update_human_readable_short_progress_log(self):
        """Create .txt file with human readable short progress log.
        Short meaning only every n-th epoch is included.
        """
        short_log = self.progress_log.take_every(20)
        last_log = self.progress_log[-1,:]
        short_log = short_log.extend(last_log)
        short_log = short_log.unique(maintain_order=True, subset=["epoch"])
        with open(f"{self.output_dir}/progress_log.txt", "w",encoding="utf-8") as f:
            print(short_log, file=f)

def save_example_images(model, output_dir):
    test_slice_ids = ["file_100", "file_200", "file_300"]
    val_slice_ids = ["file_400", "file_500", "file_600"]
    test_path = pathlib.Path(r"C:\Users\jan\Documents\python_files\adlm\data\brain\singlecoil_val") #TODO remove hardcoded paths
    val_path = pathlib.Path(r"C:\Users\jan\Documents\python_files\adlm\data\brain\singlecoil_val") #TODO remove hardcoded paths
    test_dataset = MRIDataset(test_path, specific_slice_ids=test_slice_ids)
    val_dataset = MRIDataset(val_path, specific_slice_ids=val_slice_ids)
    for slice_id in test_slice_ids:
        fully_sampled, undersampled = test_dataset.get_image(slice_id)
        with torch.no_grad():
            output = model(undersampled.unsqueeze(0))
        save_image_comparison(fully_sampled, undersampled, output, f"{output_dir}/{slice_id}_comparison.png")
    for slice_id in val_slice_ids:
        fully_sampled, undersampled = val_dataset.get_image(slice_id)
        with torch.no_grad():
            output = model(undersampled.unsqueeze(0))
        save_image_comparison(fully_sampled, undersampled, output, f"{output_dir}/{slice_id}_comparison.png")
