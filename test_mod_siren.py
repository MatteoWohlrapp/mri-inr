import torch
import pathlib
from src.networks.modulated_siren import ModulatedSiren
from src.data.mri_sampler import MRISampler
import os
from src.configuration.configuration import load_configuration, parse_args
from src.util.tiling import (
    image_to_patches,
)
from src.util.error import error_metrics


def save_args_to_file(args, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    config_path = os.path.join(output_dir, "config.txt")

    with open(config_path, "w") as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")


def test_mod_siren(config):
    print("Testing the modulated SIREN...")

    output_dir = f"{config.testing.output_dir}/{config.testing.output_name}/test"

    # Set the device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load datatset
    sampler = MRISampler(pathlib.Path(config.data.dataset), config.data.test_files)

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
        device=device,
    )

    mod_siren.load_state_dict(torch.load(config.testing.model_path))

    mod_siren.to(device)

    with torch.no_grad():
        mod_siren.eval()

        for i in range(config.data.num_samples):
            print(f"Processing sample {i + 1}/{config.data.num_samples}...")
            # Load the image
            fully_sampled_img, undersampled_img, filename = sampler.get_random_sample()

            # unsqueeze image to add batch dimension
            fully_sampled_img = fully_sampled_img.unsqueeze(0).float().to(device)
            undersampled_img = undersampled_img.unsqueeze(0).float().to(device)

            fully_sampled_patch, _ = image_to_patches(
                fully_sampled_img,
                config.model.outer_patch_size,
                config.model.inner_patch_size,
            )
            undersampled_patch, undersampled_information = image_to_patches(
                undersampled_img,
                config.model.outer_patch_size,
                config.model.inner_patch_size,
            )

            output_dir_temp = os.path.join(output_dir, filename)
            if not os.path.exists(output_dir_temp):
                os.makedirs(output_dir_temp)

            error_metrics(
                mod_siren,
                output_dir_temp,
                filename,
                fully_sampled_patch,
                undersampled_patch,
                undersampled_information,
                device,
            )


if __name__ == "__main__":
    args = parse_args()
    config_path = pathlib.Path(args.config)
    config = load_configuration(config_path, testing=True)
    test_mod_siren(config)
