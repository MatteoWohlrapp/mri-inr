import torch
import pathlib
from src.data.mri_dataset import MRIDataset
from src.networks.modulated_siren import ModulatedSiren
from src.train.training import Trainer
from src.util.util import time_function
from src.configuration.configuration import load_configuration, parse_args


@time_function
def train_mod_siren(config):
    print("Training the modulated SIREN...")

    # Load dataset
    train_dataset = MRIDataset(
        pathlib.Path(config.data.train.dataset),
        number_of_samples=config.data.train.num_samples,
    )

    val_dataset = (
        MRIDataset(
            pathlib.Path(config.data.val.dataset),
            number_of_samples=config.data.val.num_samples,
            outer_patch_size=config.model.outer_patch_size,
            inner_patch_size=config.model.inner_patch_size,
        )
        if config.data.val.dataset
        else None
    )

    # Set the device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load the model
    mod_siren = ModulatedSiren(
        dim_in=config.model.dim_in,
        dim_hidden=config.model.dim_hidden,
        dim_out=config.model.dim_out,
        num_layers=config.model.num_layers,
        latent_dim=config.model.latent_dim,
        w0=config.model.w0,
        w0_initial=config.model.w0_initial,
        use_bias=config.model.use_bias,
        dropout=config.model.dropout,
        modulate=True,
        encoder_type=config.model.encoder_type,
        encoder_path=config.model.encoder_path,
        outer_patch_size=config.model.outer_patch_size,
        inner_patch_size=config.model.inner_patch_size,
        siren_patch_size=config.model.siren_patch_size,
        device=device,
    )
    mod_siren.to(device)

    # Define the criterion and optimizer
    if config.training.optimizer == "Adam":
        optimizer = torch.optim.Adam(mod_siren.parameters(), lr=config.training.lr)
    elif config.training.optimizer == "SGD":
        optimizer = torch.optim.SGD(mod_siren.parameters(), lr=config.training.lr)
    else:
        raise ValueError("Unsupported optimizer")

    # Define the trainer
    trainer = Trainer(
        model=mod_siren,
        device=device,
        train_dataset=train_dataset,
        optimizer=optimizer,
        output_dir=config.training.output_dir,
        val_dataset=val_dataset,
        batch_size=config.training.batch_size,
        output_name=config.training.output_name,
        outer_patch_size=config.model.outer_patch_size,
        inner_patch_size=config.model.inner_patch_size,
        siren_patch_size=config.model.siren_patch_size,
        save_interval=config.training.save_interval,
        num_workers=config.data.train.num_workers,
    )

    # Check if we want to load an existing model
    if config.training.model.continue_training:
        trainer.load_model(
            model_path=pathlib.Path(config.model.model_path),
        )

    # Train the model
    trainer.train(config.training.epochs)


if __name__ == "__main__":
    args = parse_args()
    config_path = pathlib.Path(args.config)
    config = load_configuration(config_path)
    train_mod_siren(config)
